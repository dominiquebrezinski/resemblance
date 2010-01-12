# copywrite 2010 Dominique Brezinski All Rights Reserved
require 'yaml'
require 'rubygems'
require 'ai4r'

module Resemblance
  NGRAPH    = 'ngraph'
  NGRAM     = 'ngram'
  DEFAULT_N = 3
  
  def self.generate_testable_corpus(corpus_nvp, options = {})
    corpus_nvp.inject({}) do |h, values|
      name, value = *values
      h[name] = generate_n_set(value, options)
      h
    end
  end
  
  def self.generate_n_set(data, options = {})
    prefs = {:evaluation_method => NGRAPH}.merge(options)
    begin
      case prefs[:evaluation_method]
        when NGRAPH
          Resemblance::NGraphSet.new(data, options)
        when NGRAM
          Resemblance::NGramSet.new(data, options)
        else
          Resemblance::NGraphSet.new(data, options)
      end
    rescue => e
      puts "#{name}: #{value}"
      raise e
    end
  end
  
  def self.evaluate_text_against_corpus(text, corpus = {}, options = {})
    text_n_set = generate_n_set(text, options)
    r_max = 0.0
    results = generate_testable_corpus(corpus, options).collect do |entry|
      name, n_set = *entry
      # switch to longer sequence length if both texts are sufficently long
      if (options[:evaluation_method] != NGRAM) && (text_n_set.data_size > 1200 && n_set.data_size > 900)
        options[:n] = 4
      end
      r = text_n_set.r(n_set, options[:n] || DEFAULT_N)
      r_max = r if r > r_max
      [name, r]
    end
    [r_max, results]
  end
  
  def self.clustered_results_over_r_threshold(results, r_threshold)
    cluster = Ai4r::Clusterers::KMeans.new
    data_set = Ai4r::Data::DataSet.new.set_data_items(results).set_data_labels(['name', 'r'])
    cluster.build(data_set, 5)
    centroids = cluster.centroids
    #largest_r_cluster = centroids.index(centroids.max {|a,b| a.last <=> b.last})
    #largest_centroid_r_value = centroids[largest_r_cluster].last
    results.select do |nv|
      centroids[cluster.eval(nv)].last > r_threshold
    end
  end
  
  class NGraphSet
    def initialize(data, options = {})
      @options = {:character_removal_set => /[^a-zA-Z0-9 ]/, 
                  :token_separator => /\s*/, 
                  :n => DEFAULT_N}.merge(options)
      @data = data
      @n_length_sets = {@options[:n] => generate_ngraph_set(generate_words(@data,
                                                                           @options[:character_removal_set],
                                                                           @options[:token_separator]),
                                                                           @options[:n]) }
    end
    
    def intersection(compare_to, n = DEFAULT_N)
      set_intersection(self.set_where_n_is(n), compare_to.set_where_n_is(n))
    end
    
    def union(compare_to, n = DEFAULT_N)
      set_union(self.set_where_n_is(n), compare_to.set_where_n_is(n))
    end
    
    def r(compare_to, n = DEFAULT_N)
      calculate_resemblance(self.set_where_n_is(n), compare_to.set_where_n_is(n))
    end
    
    def set_where_n_is(n)
      @n_length_sets[n] ||= generate_ngraph_set(generate_words(@data, @options[:character_removal_set], @options[:token_separator]), n)
    end
    
    def data
      @data
    end
    
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
  
  class NGramSet < NGraphSet
    def initialize(data, options = {}) 
      super(data, {:token_separator => /\s+/}.merge(options))
    end
  end
 
end

yaml_data = File.open(ARGV[0], 'r').read

profile_data = YAML::load(yaml_data)

profile_data.each do |profile_name, text|
  r_max, results = Resemblance::evaluate_text_against_corpus(text, profile_data.reject{|k,v| k == profile_name})
  Resemblance::clustered_results_over_r_threshold(results, 0.2).each do |nv|
    name, r = *nv
    puts "%s <%.3f> %s" % [profile_name, r, name]
  end
end