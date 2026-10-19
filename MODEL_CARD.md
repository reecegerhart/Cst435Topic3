# Model Card: See Sense

## Model Details

See Sense is a small convolutional neural network (CNN) built with PyTorch to classify images into four categories. The model uses two convolutional layers with ReLU activation and max pooling before passing the results to a linear classifier. It is trained using the Adam optimizer and cross entropy loss.

The model also uses Grad CAM to help explain its predictions. Grad CAM creates a heatmap showing which areas of an image had the most influence on the model's prediction. This gives users a way to see what the model focused on instead of only seeing its predicted class and confidence score. However, the heatmap does not guarantee that the model made the correct prediction.

Training settings, including learning rate, batch size, and number of epochs, are provided through the API for each run.

**Model Owner:** Add team member names here.

## Intended Use

The main purpose of See Sense is to demonstrate how an image classification model can be trained, evaluated, deployed, and explained using multiple cloud services. Users can train the model with different settings, review its performance, and use Grad CAM to examine its predictions.

This project is intended for learning and demonstration purposes only. It should not be used for real world applications such as medical imaging, surveillance, or content moderation. The model is trained on synthetic geometric shapes rather than real world images.

## Data

The model uses synthetic 28 by 28 pixel grayscale images containing one of four shapes: horizontal bars, vertical bars, diagonal lines, and filled blocks. Random pixel noise is added to make the images less uniform.

Images are generated using the `shared/data.generate_images` function. A seed allows the dataset to be generated consistently when the same settings are used. The dataset configuration, including the number of images, image size, noise level, and seed, is stored in the Supabase `datasets` table. The generated image pixels are not stored in that table.

No real images or personal information are used to train the model.

## Model Performance

The model is evaluated using a held out test set containing 20% of the dataset. Two metrics are reported: accuracy and macro F1 score. Accuracy measures the percentage of predictions that are correct, while macro F1 measures performance across all four classes by giving each class equal weight.

These metrics are calculated by the API without requiring scikit learn. Each training run's results are saved in the Supabase `runs` table. The interface reads these saved results rather than calculating the metrics again, keeping the stored results as the main source of truth.

Actual performance will depend on the training settings and the generated dataset. The model card should be updated with representative results from completed training runs.

## Privacy and Data Storage

See Sense is designed to avoid storing raw prediction images. When an image is submitted for prediction, the system stores its SHA256 hash, width, height, predicted class, and confidence score in the Supabase `image_metadata` table. The original image pixels are not stored in this table.

The trained model is stored as a base64 encoded, pickled artifact in the `run_artifacts` table. This table is not readable through the anonymous client access used by the interface. The interface's anonymous key is limited to accessing run metrics.

These storage restrictions help reduce the amount of image data retained by the application. However, a hash should not be treated as complete anonymity, and the security of the stored model also depends on the database permissions and API configuration.

## Limitations

See Sense is a relatively small CNN trained on simple geometric shapes. It is not designed to recognize objects in photographs or handle real world image classification tasks.

Grad CAM provides a general indication of which image regions influenced a prediction. It does not show exactly how the model reached its decision, and the heatmap can be misleading, especially when the model has not been trained sufficiently.

Model performance is also affected by the learning rate, batch size, number of epochs, and noise in the dataset. For example, training for too few epochs may result in underfitting, which can reduce classification accuracy and produce less useful heatmaps.

Because the images are generated from known rules, strong test results do not demonstrate that the model would perform well on real world images.

## Ethical Considerations

See Sense should be presented honestly as an educational demonstration rather than a production ready image classification system. Users should be able to review the model's actual evaluation results and understand the limitations of both its predictions and explanations.

Grad CAM can make the model easier to understand, but a convincing heatmap does not prove that a prediction is correct or that the model is relying on the right features. Reporting performance metrics alongside the heatmaps helps users evaluate the model more critically.

The project also follows a data minimization approach by avoiding storage of raw prediction images. These safeguards support responsible use, but the application still depends on properly configured access controls and secure handling of stored model artifacts.