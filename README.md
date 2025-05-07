# sensuser
A simple "toy" multilayer perceptron (MLP).
This is something that has been around for a long time.
See: https://en.wikipedia.org/wiki/Multilayer_perceptron


sensuser is a QT application used to create and edit image classifier models that can be used with libnoodlenet found at: https://github.com/gnaservicesinc/libnoodlenet

By “toy," the model is designed to be small, the implementation simplistic, and easy to dig into and understand. (This started as a simple single-layer thing; it has gotten more complex, but it is still very simple.)

Usage:

Click on the button labeled "load" next to the label "Positive Examples:".

Open a folder with one or more 512x512 PNG images with the object or pattern you want to train the model to identify.

Click "Train Model."

When it is finished, you may export the model by clicking "export model."

You can import a model and do more training and/ or testing with it.

You can use exported models in C and C++ projects with libnoodlenet found at: https://github.com/gnaservicesinc/libnoodlenet

If you add a negative examples folder, you can evaluate the accuracy of the model.

The negative examples should contain one or more 512x512 PNG images that DO NOT include the object or pattern you want to train the model to identify.


You can tweak settings to suit your needs.

Note:

Be careful with the number of hidden neurons; you can make the RAM requirements skyrocket into the hundreds of GB really easily with it; keep it low. The default is 128, which works well on my system with 64gb of RAM.
