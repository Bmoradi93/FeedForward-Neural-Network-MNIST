from modules import mnist_ff_net

if __name__ == "__main__":

    # Defining an object form mnist_ff_net
    ff_net = mnist_ff_net()

    # Loading dataset
    ff_net.load_mnist_dataset()

    # Dataset Preprocessing
    ff_net.process_dataset()

    # Defining the model
    ff_net.create_ff_model()

    # Training
    ff_net.train()

    # Testing
    ff_net.test()

    # Plotting results
    ff_net.plot_results()

