from kfp import dsl, compiler
from kfp.dsl import Input, Output, Artifact, Dataset
import numpy as np

EXPERIMENT_NAME = 'Boston-house-pred'

@dsl.component(base_image='python:3.9', packages_to_install=["google-cloud-storage", "pandas"])
def load_dataset_from_gcs(bucket_name: str, blob_name: str, output_dataset: Output[Dataset]): 
    from google.cloud import storage
    import pandas as pd
    from io import StringIO
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    dataset_content = blob.download_as_string().decode('utf-8')

    data = pd.read_csv(StringIO(dataset_content), header=None, delim_whitespace=True)
    data.to_csv(output_dataset.path, header=True, index=False)

@dsl.component(base_image='python:3.9', packages_to_install=["pandas"])
def preprocess_the_dataset(dataset_content: Input[Dataset], out_data: Output[Dataset]):
    import pandas as pd
    data = pd.read_csv(dataset_content.path, header=0)
    if data.isna().sum().any():
        raise ValueError("The data needs preprocessing (remove missing values)")
    
    data.to_csv(out_data.path, index=False)

@dsl.component(base_image='python:3.9', packages_to_install=["scikit-learn", "pandas"])
def train_test_split(input_df: Input[Dataset], 
                     X_train_artifact: Output[Dataset], 
                     X_test_artifact: Output[Dataset], 
                     y_train_artifact: Output[Dataset], 
                     y_test_artifact: Output[Dataset]):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    df = pd.read_csv(input_df.path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train.to_csv(X_train_artifact.path, index=False)
    X_test.to_csv(X_test_artifact.path, index=False)
    y_train.to_csv(y_train_artifact.path, index=False)
    y_test.to_csv(y_test_artifact.path, index=False)

@dsl.component(base_image='python:3.9', packages_to_install=['numpy', 'scikit-learn', 'joblib', "pandas", "google-cloud-storage"])
def model_training(X_train_input: Input[Dataset],
                   X_test_input: Input[Dataset],
                   y_train_input: Input[Dataset],
                   X_test_scaled: Output[Dataset],
                   model_output: Output[Artifact]):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    import joblib
    from google.cloud import storage
    import pandas as pd
    scaler = StandardScaler()

    X_train = pd.read_csv(X_train_input.path)
    X_test = pd.read_csv(X_test_input.path)
    y_train = pd.read_csv(y_train_input.path)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled2= pd.DataFrame(scaler.transform(X_test))
    X_test_scaled2.to_csv(X_test_scaled.path, index=False)  # Fixing typo here

    regression = LinearRegression()
    regression.fit(X_train_scaled, y_train)

    model_file = '/trained_model.joblib'
    joblib.dump(regression, model_file)
    # Upload the model file to Google Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket('boston-house-data')
    blob = bucket.blob('data/model.pkl')
    blob.upload_from_filename(model_file)
    model_output.file = model_file

@dsl.component(base_image='python:3.9', packages_to_install=["pandas", "joblib","google-cloud-storage","scikit-learn"])
def predict(X_test: Input[Dataset], trained_model: Input[Artifact], prediction: Output[Dataset]):
    import joblib
    import pandas as pd
    from google.cloud import storage
    import sklearn
    X_test_data = pd.read_csv(X_test.path)

    storage_client = storage.Client()
    bucket = storage_client.bucket("boston-house-data")
    blob = bucket.blob("data/model.pkl")
    model_file = 'model.pkl'
    blob.download_to_filename(model_file)

    
    regression = joblib.load(model_file)

    predictions = regression.predict(X_test_data)
    pd.DataFrame(predictions).to_csv(prediction.path, index=False)

@dsl.component(base_image='python:3.9', packages_to_install=["pandas", "scikit-learn", "numpy"])
def evaluate(y_test: Input[Dataset], predictions: Input[Dataset], metrics_output: Output[Artifact]):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import pandas as pd
    import numpy as np
    y_test_data = pd.read_csv(y_test.path)
    predictions_data = pd.read_csv(predictions.path)

    mae = mean_absolute_error(y_test_data, predictions_data)
    mse = mean_squared_error(y_test_data, predictions_data)
    rmse = np.sqrt(mse)

    with open(metrics_output.path, 'w') as f:
        f.write(f'MAE: {mae}\n')
        f.write(f'MSE: {mse}\n')
        f.write(f'RMSE: {rmse}\n')

@dsl.pipeline(
    name="Boston-house-training-prediction",
    description='A pipeline to prepare dataset, split into train and test sets, train a model, and predict',
    pipeline_root='gs://boston-house-pred'
)
def pipeline():
    read_data = load_dataset_from_gcs(bucket_name="boston-house-data", blob_name="data/housing.csv")
    preprocess_data = preprocess_the_dataset(dataset_content=read_data.outputs['output_dataset'])
    split = train_test_split(input_df=preprocess_data.outputs['out_data'])
    trained_model = model_training(X_train_input=split.outputs['X_train_artifact'],
                                   X_test_input=split.outputs['X_test_artifact'],
                                   y_train_input=split.outputs['y_train_artifact']
                                   )  # Fixed typo here
    predicted_value = predict(X_test=trained_model.outputs['X_test_scaled'], trained_model=trained_model.outputs['model_output'])
    evaluate(y_test= split.outputs['y_test_artifact'], predictions=predicted_value.outputs['prediction'])  # Fixed typo here

    

pipeline_file = 'components_pipeline.yaml'
compiler.Compiler().compile(pipeline_func=pipeline, package_path=pipeline_file)
