# Group 1 Project
## Skin Lesion Detection

### What topic did we choose and why did we choose it?

Skin cancer detection based on lesion pictures is the topic we chose. Most people are affected in some way and at some point in their life by skin cancer or the question of if it’s cancer. Most people have a decent amount of sun exposure without the protection of sunscreen. Our objective is to train a model to analyze images of lesions, and accurately predict whether a lesion has the properties of cancer and if the person should seek medical follow-up.

### Slide show link

https://docs.google.com/presentation/d/1pEROkfh9EN2Nmx_0xZguhjlHKjxNGgsoeJCavWnAmtA/edit?usp=sharing


### Data Source
Tschandl, Philipp, 2018, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions", https://doi.org/10.7910/DVN/DBW86T, Harvard Dataverse, V3, UNF:6:/APKSsDGVDhwPBWzsStU5A== [fileUNF]

### Resources

- Visual Data Storage: Amazon S3
- Database: PostgreSQL
- Languages: SQL, Python

### Database Utilization

The dataset includes a metadata spreadsheet and four spreadsheets containing pixel counts for the sample images.
  - ![image](https://user-images.githubusercontent.com/83254435/133031814-8f12f8e6-ceb9-41ed-ad26-154fc2361f18.png)
  - ![image](https://user-images.githubusercontent.com/83254435/133032037-62079814-8191-4fb4-af83-dcf6787781c4.png)

The four spreadsheets can be joined into a single document. For the sake of space, the ERD does not display all 785 columns for these csv files. However, the naming structure remains consistent throughout.
  - ![ERD](https://user-images.githubusercontent.com/83254435/133032192-a65346b6-0e47-4d41-9100-db98e9fc2aa2.png)
  - "image_id" primary key will correspond with the sample images used to train the model
    - ![image](https://user-images.githubusercontent.com/83254435/133032897-c36a3ad6-4f19-4b90-a20b-4151f02566e7.png)

 
### ML Model

Data Pre-Processing:
  - A filtered metadata spreadsheet was created. The dx_type column was dropped, and categories within the dx and localization columns were condensed into functional groupings.
    - ![image](https://user-images.githubusercontent.com/83254435/135788982-1227adab-4422-4b45-b7a2-b032f0a79dab.png)
    - mel (melanoma) and bcc (basal cell carcinoma) were combined into a Cancer category, nv (melanocytic nevi) was renamed with the common name Mole, and the remaining non-cancerous lesion types were combined into an Other category.
    - ![image](https://user-images.githubusercontent.com/83254435/135789380-25233c4d-c186-47c4-bd9e-4e5672c7a69c.png)
    - ![image](https://user-images.githubusercontent.com/83254435/135789404-4beaf007-e9f5-48bb-a547-7d6a18dbb4ba.png)

  - localization categories were reduced, with lower frequency localizations grouped into an Other category.
    - ![image](https://user-images.githubusercontent.com/83254435/135789755-8203bfd1-d2c4-4f61-9ea6-760068084ede.png)
    - ![image](https://user-images.githubusercontent.com/83254435/135789767-8dbe3750-e8ab-4e71-a257-7e19373c1933.png)

  - OneHotEncoder was used to encode the features into binary classifications. 
  - Data was split on the dx_Cancer feature.

Model Implementation:
  - Using keras, we created a deep neural network with two hidden layers. The first layer contained 9 neurons and the second contained 3. 
  - We used the relu activation function for all layers because our goal is to correctly identify images labeled in the Cancer category.
  - We trained the model for 100 epochs.
  - The initial model's accuracy level was 83.7%. Due to the seriousness of skin cancer, we would hope to obtain an accuracy level above 95%. 
    - To improve accuracy, we will try training for longer, and will look at the neurons per layer to see if it needs more. 
    - For comparison, it may be helpful to perform a Logistic Regression as an alternative to the Neural Network.
