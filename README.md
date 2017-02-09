
Source code :
    src/
        datasets.py - Different datasets classes that share common interface, it loads data and provides generators
        to iterate over minibatches
        model_***.py - Classes that implement AAE architectures. Dense neural networks, convolution neural networks,
        networks that use Subpix layers.
        aae_**_solver.py - Classes that implement different training procedures. Pixel Matching AAE, Feature matching
        AAE using GAN network
        utils.py - Functions providing higher level abstraction over pure tensorflow ops.
    train_aae.py - Training procedure with different scenarios
    interface.py - Little GUI program, used for sampling from models and traversing latent space
    draw_samples.py - Visualize samples from input encoded into latent representation
    draw_images_**.py - Scripts to generate images from models produced during training.
    create_*.py - Scripts used to create datasets