# If you have OpenSSL installed, we recommend updating
# the following line to use "https"
source 'http://rubygems.org'

gem "middleman", "~>3.1.6"

# Live-reloading plugin
gem "middleman-livereload", "~> 3.1.0"

# For faster file watcher updates on Windows:
gem "wdm", "~> 0.1.0", :platforms => [:mswin, :mingw]

# Cross-templating language block fix for Ruby 1.8
platforms :mri_18 do
  gem "ruby18_source_location"
end

# Fast deploying to Github
gem "middleman-deploy"

# GitHub flavored markdown engine.
gem "redcarpet", "~> 3.1.0"

# Graph library
gem "nvd3-rails", :path => "lib/nvd3-rails", :require => false

# Icons!
gem "ionicons-rails", "~> 1.4.1.0", :require => false