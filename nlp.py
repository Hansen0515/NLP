import tensorflow as tf
import pandas
from transformers import BertTokenizer, TFBertForSequenceClassification

train_data = pandas.read_csv("train.csv")
test_data = pandas.read_csv("test.csv")

train_data.fillna("0", inplace = True)
test_data.fillna("0", inplace = True)
train_label = train_data.pop("target")

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

train_tokenized = tokenizer((train_data["text"] + "["+ train_data["keyword"]+"]" +train_data["location"]).to_list(),
                            truncation=True,
                            padding=True)
test_tokenized = tokenizer((test_data["text"] + "["+ test_data["keyword"] +"]" + test_data["location"]).to_list(),
                           truncation=True,
                           padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_tokenized),
    train_labels
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_tokenized),
))


# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.fit(train_dataset.shuffle(1000).batch(32),
#           epochs=3,
#           batch_size=32)
#
# predictions = model.predict(test_dataset.batch(32))
#
# result = pandas.read_csv("sample_submission.csv")
# result["target"] = tf.argmax(predictions.logits, axis=1).numpy()

