require 'yaml'
require 'resemblance'

yaml_data = File.open(ARGV[0], 'r').read

profile_data = YAML::load(yaml_data)

profile_data.each do |profile_name, text|
  r_max, results = Resemblance::evaluate_text_against_corpus(text, profile_data.reject{|k,v| k == profile_name}, {:use_variable_n => true})
  cluster = Resemblance::ResultsClusterer.new(results)
  cluster.results_over_r_threshold(0.2).each do |nv|
    name, r = *nv
    puts "%s <%.3f> %s" % [profile_name, r, name]
  end
  puts "%s Min R: %.3f Max R: %.3f" % ([profile_name] + cluster.min_and_max_r)
end
