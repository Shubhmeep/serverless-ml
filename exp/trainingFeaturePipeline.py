from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import hopsworks
import os
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema


# creating a feature view from feature group
project = hopsworks.login()
fs = project.get_feature_store()

try: 
    feature_view = fs.get_feature_view(name="iris", version=1)
except:
    iris_fg = fs.get_feature_group(name="iris", version=1)
    query = iris_fg.select_all()
    feature_view = fs.create_feature_view(name="iris",
                                      version=1,
                                      description="Read from Iris flower dataset",
                                      labels=["variety"],
                                      query=query)
    
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
print(X_train.shape,X_test.shape)

model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)

metrics = accuracy_score(y_test, y_pred)
print(metrics)


# creating model registry code
mr = project.get_model_registry()

model_dir="iris_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

joblib.dump(model, model_dir + "/iris_model.pkl")


input_example = X_train.sample()
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)


iris_model = mr.python.create_model(
    name="iris", 
    metrics={"accuracy" : metrics},
    model_schema=model_schema,
    input_example=input_example, 
    description="Iris Flower Predictor")

iris_model.save(model_dir)
