# %% [markdown]
# ### Course: Machine Learning
# 
# ### Date: 26th January 2024
# 
# ### Asignment: Lab1

# %% [markdown]
# ## Question 1

# %%
import pandas as pd
import numpy as np
import torch as tc
import zipfile


# %%
# Extract the contents of the 'archive.zip' file into the 'extracts' directory
with zipfile.ZipFile("archive.zip", "r") as z:
    z.extractall("extracts")

# Read the CSV file 'car_web_scraped_dataset.csv' into a pandas DataFrame
car_dataframe = pd.read_csv("extracts/car_web_scraped_dataset.csv")


# %%
car_dataframe.shape

# %%
car_dataframe.head(5)

# %% [markdown]
# ## Question 2

# %% [markdown]
# - This is suitable for regression because in regression, the goal is to understand the relationship between one or more independent variables and the dependent(target) variable which in this case could be the price which is also a continuous numeric outcome. The given dataset includes a numeric target variable, "price," and offers the opportunity to explore and quantify the relationships between various independent variables such as "year," "miles," "color," and "condition" in predicting the car prices.

# %% [markdown]
# ## Question 3

# %% [markdown]
# <ol type="a">
#   <li>
#   <h3>Approach Explanation</h3>
#     To determine the appropriate price category for each car in the dataset, I opted for a quartile-based approach. By calculating the first, second, and third quartiles of the 'price' column, I identified distinct price ranges. Cars with prices falling below the first quartile are categorized as 'cheap,' those between the first and second quartiles as 'average,' those between the second and third quartiles as 'expensive,' and those exceeding the third quartile as 'very expensive.' This strategy is designed to offer a meaningful classification, taking into account the statistical distribution of car prices in the dataset.
#   </li>
# 
#  
#  </br>
#   <li>
#   <h3>Approach Implementation</h3>
#   
#   </li>
# </ol>

# %%
car_dataframe['price'].describe()

# %%
# Convert the 'price' column to numeric after removing any '$' and ',' characters
car_dataframe['price'] = pd.to_numeric(car_dataframe['price'].replace('[\$,]','', regex=True), errors='coerce')

# Calculate the quartiles of the 'price' column
quantiles = car_dataframe['price'].quantile(q=[0.25, 0.5, 0.75, 1])
quantiles

# %%
quantiles=quantiles.to_numpy()

# %%
def group_price(price):
    """
    Group the price into categories based on quantiles.
    Args:
    price (float): The price to be categorized.
    Returns:
    str: The category of the price.
    """
    if(price <= quantiles[0]):
        return 'cheap'
    elif(quantiles[0] < price <= quantiles[1]):
        return 'average'
    elif(quantiles[1] < price <= quantiles[2]):
        return 'expensive'
    elif(price > quantiles[2]):
        return 'very expensive'

# %%
car_dataframe['price_category'] = car_dataframe['price'].apply(group_price)

# %%
car_dataframe.shape

# %%
car_dataframe.head()

# %% [markdown]
# ## Question 4

# %% [markdown]
# *All the categorical features were preprocessed into numerical values using one-hot encoding*
# After the target variable **price** was being predicted by the linear regression model using the other features in the dataset together with the encoded features.

# %%
car_dataframe.dtypes

# %%
car_dataframe.dropna(inplace=True)

# %%

# One-hot encoding categorical columns to numerical values
car_dataframe = pd.get_dummies(car_dataframe, columns=['condition', 'color', 'price_category', 'name'], drop_first=True)

# Converting 'miles' column to numeric, removing non-numeric characters
car_dataframe['miles'] = pd.to_numeric(car_dataframe['miles'].replace('[\D,]','', regex=True ), errors='coerce')
car_dataframe.head()

# %%
# Converting the car_dataframe to float32 data type
car_dataframe = car_dataframe.astype('float32')
car_dataframe.head()

# %%
car_dataframe.dtypes

# %%
# Converting DataFrame columns to PyTorch tensors and  drop the price column as the labels
inputs = tc.tensor(car_dataframe.drop('price', axis=1).values, dtype=tc.float32)
outputs = tc.tensor(car_dataframe['price'].values, dtype=tc.float32)

# %% [markdown]
# 

# %% [markdown]
# <ol type='a'>
#     <li>The input and the target tensors are</li>
# 
# <ol>

# %%
print(inputs)

# %%
print(outputs)

# %% [markdown]
# b. 

# %%
def generate_random_params(num_params):
    """
    Generate random parameters with the specified number of parameters.
    Args:
    num_params (int): The number of parameters to generate.
    Returns:
    torch.Tensor: Randomly generated parameters with the specified number of parameters.
    """
    weights = tc.rand((num_params, 1), requires_grad=True)
    return weights

# %%
input_size = inputs.shape
input_size

# %%
num_params = inputs.shape[1]
random_params = generate_random_params(num_params)
print("Random parameters =  ", random_params)

# %% [markdown]
# c.

# %%
def linear_regression(inputs, weights, bias):
    """
    Performs linear regression on the given inputs using the provided weights and bias.

    Args:
    inputs (tensor): The input tensor for the regression.
    weights (tensor): The weights tensor for the regression.
    bias (tensor): The bias tensor for the regression.

    Returns:
    tensor: The result of the linear regression.
    """
    return tc.matmul(inputs, weights) + bias

# %%
def mean_squared_error(outputs, labels):
    return tc.mean((outputs - labels)**2)

# %%
predicitons = linear_regression(inputs, random_params, 0)
pd.DataFrame({'predictions': predicitons.view(-1).detach().numpy(), 'labels': outputs.view(-1).detach().numpy()})

# %%
squared_error = mean_squared_error(predicitons, outputs)
print("Mean Squared Error =  ", squared_error.item())

# %% [markdown]
# d. 

# %%
def f(x):
    """Calculates the function f(x) = 2 * x^T * x"""
    return 2 * tc.matmul(x.t(), x)

# %%


def getGMatrix(input):
    G = []
    for i in range(5):
        x = input[i]
        x.requires_grad = True
        y = f(x)
        y.backward()
        print(x.grad == 4 * x)
        print(x.grad)
        G.append(x.grad)

    return G

# %%
getGMatrix(inputs[:5, :])


