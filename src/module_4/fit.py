from custom_model import CustomModel
import json

def handler_fit(event, _) -> dict:
    '''
    Receives an event of type Dict containing the configuration for 
    loading and preprocessing data, training a model and saving it.
    '''
    input_data_local_path = event["input_data_local_path"]
    output_model_local_path = event["output_model_local_path"]
    label_col = event["label_col"]
    params = event["model_parametrisation"]
    train_cols = event["train_cols"]
    model_class = event["model_class"]

    model = CustomModel(model_class=model_class)
    model.load_data(input_data_local_path)
    model.preprocess_data()
    train_df, val_df, test_df = model.temporal_split()
    X_train, y_train = model.feature_split(train_df, label_col=label_col)
    model.train(X_train, y_train, params=params, train_features=train_cols)
    model.save_model(output_path=output_model_local_path)


    return {
        "statusCode": "200",
        "body": json.dumps({
            "message": f"Model: {model.model_name} trained and saved succesfully",
            "model_path": output_model_local_path
        })
    }

