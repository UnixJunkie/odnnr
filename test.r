
library(keras)

# # CPU version
# install_keras(method = "conda", tensorflow = "cpu")

# FBR: load data from external csv files
# FBR: take the code from oplsr R script probably

# load data
dataset <- dataset_boston_housing()

train_data <- dataset$train$x

nb_lines <- dim(train_data)[1]
nb_cols <- dim(train_data)[2]

train_targets <- dataset$train$y

test_data <- dataset$test$x

nb_cols_test <- dim(test_data)[2]
stopifnot(nb_cols == nb_cols_test)

test_targets <- dataset$test$y

# normalize data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

# define the model architecture
build_model <- function() {
    # FBR: there are 64 units in a hidden layer while the input has 13 columns ?!
    #      it already looks overdimensioned
    model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = nb_cols) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)

    # FBR: I should have R2 as loss and metrics probably
    model %>% compile(
        # optimizers 8 choices !
        # SGD
        # RMSprop
        # Adam
        # Adadelta
        # Adagrad
        # Adamax
        # Nadam
        # Ftrl
        optimizer = "rmsprop",
        # mse "mean squared error"
        # mae "mean absolute error"
        # FBR: I want (1 - R2)
        loss = "mse",
        # same choice as loss: mse or mae
        # FBR: I want R2
        metrics = c("mae")
    )
}

model <- build_model()

model %>% fit(train_data, train_targets, epochs = 50, batch_size = 1,
              verbose = 1)

model %>% evaluate(test_data, test_targets, verbose = 0)
