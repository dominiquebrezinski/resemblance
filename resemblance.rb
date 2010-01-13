#--
# Copyright 2010 Dominique Brezinski
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#++

require 'rubygems'
require 'ai4r'

# Author:: Dominique Brezinski (mailto: dominique.brezinski@gmail.com)
# Copyright:: Copyright 2010 Dominique Brezinski
# License:: GPLv3
#
# Resemblance is an implementation of Ngram and Ngraph techniques
# for identifying chunks of text with similarities. The techniques are 
# implementations and adaptions of "Detecting short passages of similar  
# text in large document collections" by Caroline Lyon, James Malcolm 
# and Bob Dickerson.
#
# While looking for a solution to detect similarity in short chunks of text,
# in the range of 20-300 words, it became clear that short (length 3 or 4) 
# Ngraphs of individual characters function better than Ngrams of words.
# Similarity in this case is meant to encompass what a human reader might
# discern as similar in style/structure/intent without resorting to syntax 
# analysis. However, for use-cases like plagiarism detection, Ngrams of 
# words is more appropriate for identifying same or similar phrases and
# sentences.
#
# The implementation provides both Ngraph and Ngram constructs to satisfy
# both use-cases.
module Resemblance
  DEFAULT_N = 3
  
  # Generate a corpus for measuring similarity against.
  # +corpus+::
  #   an enumerable of name-value pairs where the name identifies the
  #   text and value is the text itself. Example: { :the_name => "the text" }
  #   or [ ["the name", "the text"] ]
  # +options+::
  #   a hash of options, specifically the key :evaluation_method that
  #   has the corresponding value of :ngraph or :ngram. :ngraph is the 
  #   default if options are not specified. See NGraphSet or NGramSet
  #   for additional options that can be passed.
  def self.generate_testable_corpus(corpus, options = {})
    corpus.inject({}) do |h, values|
      name, value = *values
      h[name] = generate_n_set(value, options)
      h
    end
  end
  
  # Generate a set of ngraphs or ngrams for the given text. The result
  # is an instantiation of NGraphSet or NGramSet suitable for inspection
  # and comparision against other same class objects.
  # +text+::
  #    the text used to generate the N set for future comparisons and
  #    evaluation
  # +options+::
  #   a hash of options, specifically the key :evaluation_method that
  #   has the corresponding value of :ngraph or :ngram. :ngraph is the 
  #   default if options are not specified. See NGraphSet or NGramSet
  #   for additional options that can be passed.
  def self.generate_n_set(text, options = {})
    prefs = {:evaluation_method => :ngraph}.merge(options)
    case prefs[:evaluation_method]
      when :ngraph, 'ngraph'
        Resemblance::NGraphSet.new(text, options)
      when :ngram, 'ngram'
        Resemblance::NGramSet.new(text, options)
      else
        Resemblance::NGraphSet.new(text, options)
    end
  end
  
  # The highest level interface for interaction with Resemblance that
  # facilitates evaluating a text chunk against a corpus. It combines
  # generate_n_set and generate_testable_corpus and iterates over
  # the corpus evaluating text against each entry.
  # +text+::
  #    chunk of text to evaluate against the given corpus
  # +corpus+::
  #    see generate_testable_corpus for format
  # +options+::
  #   a hash of options, specifically the key :evaluation_method that
  #   has the corresponding value of :ngraph or :ngram. :ngraph is the 
  #   default if options are not specified. See NGraphSet or NGramSet
  #   for additional options that can be passed.
  #
  # Returns an array where the first entry is the maximum R value seen
  # across the corpus and the second entry is an array of arrays (result set)
  # containing the name of the corpus item and the R value for that item.
  def self.evaluate_text_against_corpus(text, corpus = {}, options = {})
    text_n_set = generate_n_set(text, options)
    r_max = 0.0
    results = generate_testable_corpus(corpus, options).collect do |entry|
      name, n_set = *entry
      # switch to longer sequence length if both texts are sufficently long
      if options[:use_variable_n] && (text_n_set.data_size > 1200 && n_set.data_size > 900)
        options[:n] = 4
      end
      r = text_n_set.r(n_set, options[:n] || DEFAULT_N)
      r_max = r if r > r_max
      [name, r]
    end
    [r_max, results]
  end
  
  # Uses K-Means clustering to segment the result set generated by
  # evaluate_text_against_corpus. The general use-case is to identify results
  # from the corpus that come near or above a certain R threshold.
  class ResultsClusterer
    # +result_set+::
    #    the result set from evaluate_text_against_corpus or any set in the
    #    form [ [name, R],...]
    # +cluster_count+::
    #    the number of clusters to segment the results into. default is 5
    def initialize(result_set, cluster_count = 5)
      if result_set.nil?
        raise ArgumentError, "result_set is nil"
      end
      @result_set = result_set
      @cluster = Ai4r::Clusterers::KMeans.new
      @data_set = Ai4r::Data::DataSet.new.set_data_items(@result_set).set_data_labels(['name', 'r'])
      @cluster.build(@data_set, cluster_count)
      @centroids = @cluster.centroids
    end
    
    # Select only the results from clusters where the centroid's R is greater
    # than the given threshold.
    # +r_threshold+::
    #    a float between 0.0 and 1.0. for short text chunks, under 300 words,
    #    and Ngraphs a value of 0.2 is a good start. when using Ngrams try
    #    0.03.
    def results_over_r_threshold(r_threshold)
      @result_set.select do |nv|
        @centroids[@cluster.eval(nv)].last > r_threshold
      end
    end
  
    # Returns an array containing the centroid's R values for the smallest
    # and largest, with regard to R, clusters.
    def min_and_max_r
      largest_r_cluster = @centroids.index(@centroids.max {|a,b| a.last <=> b.last})
      smallest_r_cluster = @centroids.index(@centroids.min {|a,b| a.last <=> b.last})
      [@centroids[smallest_r_cluster].last, @centroids[largest_r_cluster].last]
    end
  end
  
  # Encompasses the set of Ngraphs for a given text and operations over the
  # set.
  class NGraphSet
    # +text+::
    #    the text for evaluation
    # +options+::
    #    a hash of options to control text preparation, tokenization and
    #    the length of the Ngraph. :character_removal_set must be a RegExp
    #    that matches the characters to be removed from the text prior to
    #    tokenization. The default is /[^a-zA-Z0-9 ]/. :token_separator must
    #    be a String or RegExp that identifies the token (word) separator in
    #    the prepared text. The default is /\s*/ which has the effect of
    #    removing whitespace and splitting the text into individul characters.
    #    :n is the length of the Ngraphs and defaults to 3.
    def initialize(text, options = {})
      @options = {:character_removal_set => /[^a-zA-Z0-9 ]/, 
                  :token_separator => /\s*/, 
                  :n => DEFAULT_N}.merge(options)
      @data = data
      @n_length_sets = {@options[:n] => generate_ngraph_set(generate_words(@data,
                                                                           @options[:character_removal_set],
                                                                           @options[:token_separator]),
                                                                           @options[:n]) }
    end
    
    # Returns the intersection of Ngraphs between self and another instance of
    # NGraphSet.
    # +compare_to+::
    #    another instance of NGraphSet
    # +n+::
    #    length of Ngraphs to use in operation
    def intersection(compare_to, n = DEFAULT_N)
      set_intersection(self.set_where_n_is(n), compare_to.set_where_n_is(n))
    end
    
    # Returns the union of Ngraphs between self and another instance of
    # NGraphSet.
    # +compare_to+::
    #    another instance of NGraphSet
    # +n+::
    #    length of Ngraphs to use in operation
    def union(compare_to, n = DEFAULT_N)
      set_union(self.set_where_n_is(n), compare_to.set_where_n_is(n))
    end
    
    # Returns R (resemblance measure) between self and another instance of
    # NGraphSet.
    # +compare_to+::
    #    another instance of NGraphSet
    # +n+::
    #    length of Ngraphs to use in operation
    def r(compare_to, n = DEFAULT_N)
      calculate_resemblance(self.set_where_n_is(n), compare_to.set_where_n_is(n))
    end
    
    # Returns a Ngraph set using n length
    # +n+::
    #    the Ngraph length
    def set_where_n_is(n)
      @n_length_sets[n] ||= generate_ngraph_set(generate_words(@data, @options[:character_removal_set], @options[:token_separator]), n)
    end
    
    # The unaltered, original text
    def data
      @data
    end
    
    # The length, in characters, or the original text
    def data_size
      @data_size ||= @data.size
    end

   private
   
    def generate_words(data, character_removal_set = /[^a-zA-Z0-9 ]/, separator = /\s*/)
      data.gsub(character_removal_set, '').split(separator)
    end

    def generate_ngraph(word_set, n)
      word_set.size.upto(n - 1){ word_set << nil } if word_set.size < n
      word_set.join(' ')
    end

    def generate_ngraph_set(words, n)
      ngraph_set = []
      words.size.times do |i|
        ngraph_set << generate_ngraph(words[i, n], n)
      end
      ngraph_set.uniq
    end
    
    def calculate_resemblance(set_1, set_2)
      set_intersection(set_1, set_2).size.to_f / set_union(set_1, set_2).size.to_f
    end

    def set_intersection(set_1, set_2)
      set_1 & set_2
    end

    def set_union(set_1, set_2)
      (set_1 + set_2).uniq
    end
  end
  
  # Encompasses the set of Ngrams for a given text and operations over the
  # set.
  class NGramSet < NGraphSet
    # See NGraphSet ... only difference is :token_separator is /\s+/
    def initialize(text, options = {}) 
      super(data, {:token_separator => /\s+/}.merge(options))
    end
  end
 
end
