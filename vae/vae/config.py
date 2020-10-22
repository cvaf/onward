class VariationalConfig:
    # Basic data
    img_shape = (28, 28, 1)
    latent_dim = 2  # Dimensionality of latent space

    batch_size = 32
    train_size = 60000
    test_size = 10000


config = VariationalConfig()
