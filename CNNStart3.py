import pandas as pd 
import torch 
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load dataset
# -------------------------------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation",
           "relationship","race","sex","capital_gain","capital_loss","hours_per_week",
           "native_country","income"]

adult_data = pd.read_csv(URL, header=None, names=columns, skipinitialspace=True)

# Drop unused columns
adult_data.drop(['fnlwgt','education','capital_gain','capital_loss'], axis=1, inplace=True)

# Replace ? with NaN and drop missing values
adult_data.replace('?', pd.NA, inplace=True)
adult_data.dropna(inplace=True)

# -------------------------------
# 2. Encode categorical columns
# -------------------------------
Cat_columns = ['workclass','marital_status','occupation','relationship',
               'native_country','race','sex','income']

encoders = {}
for col in Cat_columns:
    le = LabelEncoder()
    adult_data[col] = le.fit_transform(adult_data[col])
    encoders[col] = le   # save encoder for later use

# -------------------------------
# 3. Prepare tensors
# -------------------------------
X = torch.tensor(adult_data.drop(['income'], axis=1).values, dtype=torch.float32)
y = torch.tensor(adult_data['income'].values, dtype=torch.float32).view(X.shape[0],1)

# -------------------------------
# 4. Define Model
# -------------------------------
class simpleNN(nn.Module):
    def __init__(self, num_neurons, out_neurons):
        super(simpleNN,self).__init__()
        self.fc1 = nn.Linear(num_neurons,5)
        self.fc2 = nn.Linear(5,4)
        self.fc3 = nn.Linear(4,3)
        self.fc4 = nn.Linear(3,2)
        self.fc5 = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,X):
        X = self.fc1(X) 
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        X = self.fc5(X)
        X = self.sigmoid(X)   # ensure output between 0 and 1
        return X

input_neurons = X.shape[1]
output_neurons = 1
model = simpleNN(input_neurons, output_neurons)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# -------------------------------
# 5. Training
# -------------------------------
num_epoch = 1000
for epoch in range(1, num_epoch+1):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:  # print every 100 epochs
        print(f'Epoch {epoch}/{num_epoch}, Loss = {loss.item():.4f}')

# -------------------------------
# 6. Prediction for new data
# -------------------------------
new_data = {
    "age": 40,
    "workclass": "Private",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hours_per_week": 50,
    "native_country": "United-States"
}

# Make dataframe for new data
new_df = pd.DataFrame([new_data])

# Apply same encoders
for col in Cat_columns[:-1]:  # exclude 'income' (target)
    new_df[col] = encoders[col].transform(new_df[col])

# Convert to tensor
X_new = torch.tensor(new_df.values, dtype=torch.float32)

# Predict
with torch.no_grad():
    prob = model(X_new)
    prediction = (prob >= 0.5).int()

print("\n--- Prediction ---")
print("Probability of income >50K:", prob.item())
print("Predicted class:", " >50K" if prediction.item()==1 else " <=50K")
