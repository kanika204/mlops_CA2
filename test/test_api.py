
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# Hard-coded sample house data
payload = {
    "MSSubClass": 20,
    "MSZoning": "RL",
    "LotFrontage": 80,
    "LotArea": 9600,
    "Street": "Pave",
    "Alley": None,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2000,
    "YearRemodAdd": 2000,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": None,
    "MasVnrArea": 0,
    "ExterQual": "Gd",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "Gd",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "GLQ",
    "BsmtFinSF1": 700,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 300,
    "TotalBsmtSF": 1000,
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": 1000,
    "2ndFlrSF": 0,
    "LowQualFinSF": 0,
    "GrLivArea": 1000,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "FullBath": 1,
    "HalfBath": 0,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "KitchenQual": "Gd",
    "TotRmsAbvGrd": 6,
    "Functional": "Typ",
    "Fireplaces": 1,
    "FireplaceQu": None,
    "GarageType": "Attchd",
    "GarageYrBlt": 2000,
    "GarageFinish": "Fin",
    "GarageCars": 2,
    "GarageArea": 500,
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 100,
    "OpenPorchSF": 50,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": None,
    "Fence": None,
    "MiscFeature": None,
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": 2010,
    "SaleType": "WD",
    "SaleCondition": "Normal"
}


# Test 1: Health endpoint
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# Test 2: Root endpoint
def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "Ready to serve"


# Test 3: Valid house price prediction
def test_valid_house_prediction():
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_price" in result
    assert result["currency"] == "USD"
    assert isinstance(result["predicted_price"], (int, float))
    assert result["predicted_price"] > 0


# Test 4: Invalid house input
def test_invalid_house_input():
    response = client.post("/predict", json={})

    assert response.status_code == 422
