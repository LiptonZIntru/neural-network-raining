import pandas as pd
import tensorflow as tf
import numpy as np

COLUMN_NAMES = ['Temperature', 'Apparent', 'Humidity', 'WindSpeed',
                'Visibility', 'Pressure']

IGNORED = ['FormattedDate', 'PrecipType', 'WindBearing', 'DailySummary', 'Cloud']

target = 'Summary'

train = pd.read_csv('csv/weatherHistory_smaller_encoded.csv')
test = pd.read_csv('csv/eval_out.csv')

train_y = train.pop(target)
test_y = test.pop(target)

for i in IGNORED:
    train.pop(i)


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))  # error, got NaN

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(len(labels)).repeat()

    return dataset.batch(batch_size)


my_feature_columns = []
for key in COLUMN_NAMES:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=14)

classifier.train(input_fn=lambda: input_fn(train, train_y, training=True),
                 steps=1000)

result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

###################################################################


OPTIONS = [
    'Partly Cloudy',  # => 0
    'Mostly Cloudy',  # => 1
    'Overcast',  # => 2
    'Breezy and Partly Cloudy',  # => 3
    'Clear',  # => 4
    'Foggy',  # => 5
    'Breezy and Mostly Cloudy',  # => 6
    'Breezy and Overcast',  # => 7
    'Humid and Mostly Cloudy',  # => 8
    'Breezy',  # => 9
    'Humid and Overcast',  # => 10
    'Drizzle',  # => 11
    'Light Rain',  # => 12
    'Rain'  # => 13
]


def input_func(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


features = COLUMN_NAMES
predict = {}

while True:
    print("Please type numeric values as prompted.")
    for feature in features:
        # valid = True
        val = input(feature + ": ")

        predict[feature] = [float(val)]

    predictions = classifier.predict(input_fn=lambda: input_func(predict))
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(
            OPTIONS[class_id], 100 * probability))
