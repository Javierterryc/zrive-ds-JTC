import joblib
import json
import pandas as pd
import os


def handler_predict(event, _) -> dict:
    '''
    Generates predictions for user data using the latest trained model.
    '''
    models_local_path = event["models_local_path"]
    train_cols = event["train_features"]
    last_model = sorted(os.listdir(models_local_path))[-1]
    model_pkl = joblib.load(os.path.join(models_local_path, last_model))
    test_user = pd.DataFrame.from_dict(event["users"], orient="index")[train_cols]

    pred_output_json = {}

    for index in test_user.index:
        user_row = test_user.loc[[index]]
        pred_arr = model_pkl.predict_proba(user_row)
        pred_output_json[index] = pred_arr[0].astype(float).tolist()


    return {
        "statusCode": "200",
        "body": json.dumps(
            {
                "message" : "Predictions generated successfuly:",
                "model_used": f"{last_model.split('/')[-1]}",
                "output" : pred_output_json
            }
        ), 
    }