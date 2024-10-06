import pandas as pd 
import numpy as np

db = pd.read_csv('restaurant_info.csv') 

# add new column with random data named (food quality, crowdedness, length of stay)
db['food_quality'] = np.random.randint(1, 10, db.shape[0])
# make crowdedness and length of stay random boolean 

db['crowdedness'] = np.random.choice([True, False], db.shape[0])
db['length_of_stay'] = np.random.choice([True, False], db.shape[0])
# save the csv as a new file 
db.to_csv('updated_restaurant_info.csv', index=False)