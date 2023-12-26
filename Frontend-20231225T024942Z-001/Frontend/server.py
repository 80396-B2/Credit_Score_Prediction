from flask import Flask, request, render_template
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the model
with open('/home/kartik/Documents/Credit_Score_Prediction/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask application
app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    # Read the file contents and send them to the client
    return render_template('index.html')


@app.route("/classify", methods=["POST"])
def classify():
    # Get the values entered by the user
    print(request.form)
    age = float(request.form.get("age"))
    annualIncome = float(request.form.get("annualIncome"))
    delayFromDueDate = float(request.form.get("delayFromDueDate"))
    numDelayedPayment = float(request.form.get("numDelayedPayment"))
    numCreditInquiries = float(request.form.get("numCreditInquiries"))
    creditMix = request.form['creditMix']
    outstandingDebt = float(request.form.get("outstandingDebt"))
    totalEMI = float(request.form.get("totalEMI"))
    creditAgeYears = float(request.form.get("creditAgeYears"))
    paymentBehaviour = request.form.getlist("paymentBehaviour")[0]
    paymentMinAmount = (request.form.get("paymentMinAmount"))

    credit_mix_val = 0
    paymentBehaviour_val = 0
    paymentMinAmount_yes = 0
    paymentMinAmount_No = 0
    paymentMinAmount_NM = 0

    # ---------- Credit mix ---------- #
    print(creditMix)
    if creditMix == "Good":
        credit_mix_val = 1
    elif creditMix == "Standard":
        credit_mix_val = 2
    else:
        credit_mix_val = 0

    # ---------- Payment Behaviour ---------- #
    if paymentBehaviour == "Low_spent_Small_value_payments":
        paymentBehaviour_val = 5
    elif paymentBehaviour == "High_spent_Medium_value_payments":
        paymentBehaviour_val = 1
    elif paymentBehaviour == "Low_spent_Medium_value_payments":
        paymentBehaviour_val = 4
    elif paymentBehaviour == "High_spent_Large_value_payments":
        paymentBehaviour_val = 0
    elif paymentBehaviour == "High_spent_Small_value_payments":
        paymentBehaviour_val = 2
    elif paymentBehaviour == "Low_spent_Large_value_payments":
        paymentBehaviour_val = 3
    else:
        paymentBehaviour_val = 6

    # ---------- PaymentMinAmount ---------- #
    if paymentMinAmount == "no":
        paymentMinAmount_No = 1
    elif paymentMinAmount == "yes":
        paymentMinAmount_yes = 1
    elif paymentMinAmount == "Not Mention":
        paymentMinAmount_NM = 1

    input_data = {
        "age": [age],
        "annualIncome": [annualIncome],
        "delayFromDueDate": [delayFromDueDate],
        "numDelayedPayment": [numDelayedPayment],
        "numCreditInquiries": [numCreditInquiries],
        "credit_mix_val": [credit_mix_val],
        "outstandingDebt": [outstandingDebt],
        "totalEMI": [totalEMI],
        "paymentBehaviour_val": [paymentBehaviour_val],
        "creditAgeYears": [creditAgeYears],
        "paymentMinAmount_NM": [paymentMinAmount_NM],
        "paymentMinAmount_No": [paymentMinAmount_No],
        "paymentMinAmount_yes": [paymentMinAmount_yes]
    }

    input_df = pd.DataFrame(input_data)
    to_scale = ["age", "annualIncome", "delayFromDueDate", "numDelayedPayment", "numCreditInquiries", "outstandingDebt", "totalEMI", "paymentBehaviour_val", "creditAgeYears"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), to_scale)
        ],
        remainder='passthrough'
    )

    # Create a pipeline with the preprocessor
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    # Fit and transform the input data
    output_df = pd.DataFrame(pipeline.fit_transform(input_df))

    # Make a prediction

    answers = model.predict(output_df.to_numpy())

    # answers = model.predict([[age, annualIncome, delayFromDueDate, numDelayedPayment,
    #                         numCreditInquiries, credit_mix_val, outstandingDebt, totalEMI
    #                         ,paymentBehaviour_val, creditAgeYears, paymentMinAmount_NM,
    #                         paymentMinAmount_No, paymentMinAmount_yes]])

    # Extract the first element of the prediction array
    prediction = int(answers[0])

    result = "Okay"

    if prediction == 0:
        result = "Good"
        # print("Good")
        # return "Good"

    elif prediction == 1:
        result = "Poor"
        # print("Poor")
        # return "Poor"

    elif prediction == 2:
        result = "Standard"
        # print("Standard")
        # return "Standard"

    return render_template("result.html", prediction_result=result)
    # return render_template("result.html", prediction_text="The credit score prediction is {}".format(result))


# Start the application
# if __name__ == "__main__":
app.run(host="0.0.0.0", port=8000, debug=True)
