pkg <- 'keras'
if (!require(pkg, character.only = TRUE)) {
  install.packages(pkg, dependencies = TRUE,
                   repos='http://cran.r-project.org')
}
library(keras)
# CPU version
install_keras(method = "conda", tensorflow = "cpu")
