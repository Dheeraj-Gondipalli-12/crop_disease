#used to create a dataset for the model, adjust the number inside the range function to generate the desired number of data values.

import pandas as pd
import random

# Lists of possible values for each column
crops = ["Wheat", "Rice", "Maize", "Soybean", "Potato", "Tomato", "Grapevine", "Apple", "Peach", "Strawberry", "Coffee", "Cocoa", "Corn", "Barley", "Oats", "Sunflower", "Peanut", "Cotton", "Sugarcane"]
diseases = ["Wheat blast", "Bacterial leaf blight", "Maize smut", "Soybean rust", "Potato late blight", "Early blight", "Powdery mildew", "Apple scab", "Peach leaf curl", "Strawberry leaf anthracnose", "Coffee leaf rust", "Cocoa black pod", "Corn rust", "Barley scald", "Sunflower rust", "Peanut smut", "Cotton wilt", "Sugarcane smut"]
temperature = [f"{random.randint(10, 40)}-{random.randint(15, 45)}" for _ in range(300)]
humidity = [f"{random.randint(40, 90)}-{random.randint(60, 100)}" for _ in range(300)]
colors = ["Brown", "Yellow", "Black", "Orange", "White", "Red", "Green", "Blue"]
impact = ["Severe", "Moderate", "Mild"]

# Generate random data
data = {
    "Crop": [random.choice(crops) for _ in range(300)],
    "Disease": [random.choice(diseases) for _ in range(300)],
    "Temperature (°C)": temperature,
    "Humidity (%)": humidity,
    "Color": [random.choice(colors) for _ in range(300)],
    "Impact": [random.choice(impact) for _ in range(300)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("<enter the filename>.csv", index=False)

