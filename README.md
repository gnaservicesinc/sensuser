## Sensuser
Sensuser is a Qt application that serves as a comprehensive workbench for creating, training, evaluating, and analyzing its own “.senm” model format. Its exported “.senm” model files can subsequently be utilized in C and C++ projects through the libnoodlenet library, available at: https://github.com/gnaservicesinc/libnoodlenet.

The “.senm” files are relatively straightforward. They begin with a JSON header containing metadata, followed by the weights, which are saved in binary format. This format is preferred over storing the actual float values as strings, as it would result in files that are several gigabytes in size. The “.senm” files are a hybrid of JSON and binary formats.

Initially conceived as a rudimentary “toy” with a single-layer perceptron capable of detecting specific shapes and employing only 32 nodes, utilizing 1-bit images (on or off, black or white pixels), Sensuser underwent a subsequent transformation into a multilayer perceptron (MLP), a concept that has been widely adopted in the field.

For further reference, please visit: https://en.wikipedia.org/wiki/Multilayer_perceptron.

As an MLP, Sensuser’s models can now be trained to accurately detect objects in images, enabling them to achieve remarkable accuracy when employing high-quality datasets and appropriate training parameters for specific scenarios.

To achieve highly accurate models, it is essential to consider the following factors:

Utilize a substantial and diverse dataset comprising numerous high-quality examples. Set the batch size to 1. (A lower batch size will yield a more precise final outcome.) Adjust the hidden neuron values to double the resolution of our training images, as they must be 512. Consequently, twice that is 1024. Indeed, there is a notable improvement in accuracy from models with 512-sized hidden neurons, particularly when comprehending more intricate features.

Reduce the learning rate and increase the number of epochs. To obtain a satisfactory model, you will require thousands of epochs on a dataset containing numerous images. If the model exhibits overfitting, lowering the learning rate should rectify this issue. (During testing, models trained in this manner yielded higher accuracy compared to those that avoided overfitting by minimizing epochs rather than lowering the learning rate.)

Implement these adjustments prior to commencing model training. Training will take significantly longer, necessitate substantial RAM usage, and result in a more accurate model. While you can utilize as few as 10 or 20 images, the resulting model will lack robustness and reliability.

To compromise accuracy for expedited model training, reduce RAM usage, minimize resource consumption, and significantly decrease model file sizes. Adjust the hidden neuron values to less than 512, such as 256 or 128. You can also reduce the number of images.

Despite its growing scale, Sensuser remains designed as a prototype for experimentation with machine learning. Consequently, the focus remains on a remarkably simple and compact project, preserving its intended simplicity. Designed as a “toy,” Sensuser prioritizes user-friendliness, ease of use, and intuitive understanding. Its implementation is straightforward, facilitating effortless exploration and comprehension.

Usage:

1. Select the “load” button adjacent to the “Positive Examples: “ label.
2. Navigate to a folder containing one or more 512x512 PNG images representing the object or pattern for which you intend to train the model.
3. Proceed with training the model by clicking “Train Model.”
4. Upon completion, export the model by selecting “export model.”
5. Import the model for further training and testing.
**6. Utilize exported models in C and C++ projects via the libnoodlenet library, accessible at:** https://github.com/gnaservicesinc/libnoodlenet.
7. Optionally, create a negative examples folder to evaluate the model’s accuracy.

Negative examples should consist of one or more 512x512 PNG images that do not contain the object or pattern for which the model is being trained.

Settings:

Sensuser offers customizable settings to accommodate your specific requirements.




Future Enhancements [TODO]:

	⁃	Support for more hidden layers, configurable by the user.
	⁃	Different activation functions (Tanh, ReLU, Leaky ReLU).
	⁃	Different optimization algorithms (Adam, RMSprop).
	⁃	Regularization techniques (L1, L2, Dropout) to prevent overfitting.
	⁃	Data augmentation for images (rotation, flipping, small translations).
	⁃	Saving/loading training progress (checkpoints).
	⁃	More detailed performance metrics (precision, recall, F1-score, ROC curve).
	⁃	Implement better parallelization multithreading in the sensor to increase the training speed of the sensor.
