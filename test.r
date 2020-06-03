
library(keras)

# CPU version
install_keras(method = "conda", tensorflow = "cpu")

# # GPU version
# install_keras(method = "conda", tensorflow = "gpu")

dataset <- dataset_boston_housing()

train_data <- dataset$train$x
train_targets <- dataset$train$y

test_data <- dataset$test$x
test_targets <- dataset$test$y

# normalize data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

# define the model architecture
build_model <- function() {
    model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[[2]]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)

    # FBR: I should have R2 as loss and metrics probably
    model %>% compile(
        optimizer = "rmsprop",
        loss = "mse",
        metrics = c("mae")
    )
}

model <- build_model()

model %>% fit(train_data, train_targets, epochs = 50, batch_size = 1,
              verbose = 1)

model %>% evaluate(test_data, test_targets, verbose = 0)
