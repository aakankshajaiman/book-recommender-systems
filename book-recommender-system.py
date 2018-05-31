import pandas as pd
import numpy as np
# import the functions for cosine distance, euclidean distance
from scipy.spatial.distance import cosine, euclidean, correlation
from sklearn.model_selection import train_test_split

# #### Examining the unique users, and unique items
def uniqueData(dense_matrix,df_data1):
    total = dense_matrix.size
    missingCount = dense_matrix.isnull().sum().sum()
    per = (missingCount * 100)/total
    print "Unique Users: %d" % len(df_data1["user"].unique())
    print "Unique Books: %d" % len(df_data1["item"].unique())
    print "Total Values : %d" % total
    print "Missing Values : %d" % missingCount
    print "Percentage : %d" % per



def correlationCheck (dense_matrix):
    dense_matrix = dense_matrix.fillna(0)
    dense_matrix.head()
    print correlation(dense_matrix.loc['2'].astype(np.int64), dense_matrix.loc['3'].astype(np.int64))


# define a functions, which takes the given item as the input and returns the top K similar items (in a data frame)
def top_k_items(dense_matrix, item_number, k):
    dense_matrix = dense_matrix.fillna(0)
    dense_matrix.head()

    # copy the dense matrix and transpose it so each row represents an item
    df_sim = dense_matrix.transpose()
    # remove the active item
    df_sim = df_sim.loc[df_sim.index != item_number]
    df_sim = df_sim.loc[df_sim.index != 'itemID']
    item_number = str(item_number)
    print 'Euclidean Distance of 1 with 18: %f' % euclidean(dense_matrix['1'].astype(np.int64), dense_matrix['18'].astype(np.int64))
    print 'Euclidean Distance of 1 with 36: %f' % euclidean(dense_matrix['1'].astype(np.int64), dense_matrix['36'].astype(np.int64))

    # calculate the distance between the given item for each row (apply the function to each row if axis = 1)
    df_sim["distance"] = df_sim.apply(lambda x: euclidean(dense_matrix[item_number].astype(np.int), x.astype(np.int)), axis=1)
    # return the top k from the sorted distances
    return df_sim.sort_values(by="distance").head(k)["distance"]


# define a functions, which takes the given user as the input and returns the top K similar users (in a data frame)
def top_k_users(dense_matrix,user_number, k):
    # no need to transpose the matrix this time because the rows already represent users
    # remove the active user
    df_sim = dense_matrix.loc[dense_matrix.index != user_number]
    # calculate the distance for between the given user and each row
    df_sim["distance"] = df_sim.apply(lambda x: euclidean(dense_matrix.loc[user_number], x), axis=1)
    # return the top k from the sorted distances
    return df_sim.sort_values(by="distance").head(k)["distance"]



def user_based_predict(df_train_x, df_test_x, df_train_y, user_number, item_number, k=10):
    # copy from all the training predictors
    df_sim = df_train_x.copy()
    user_number = str(user_number)
    df_sim = df_sim.loc[df_sim.index != 'userID']

    # for each user, calculate the distance between this user and the active user
    df_sim["distance"] = df_sim.apply(lambda x: euclidean(df_test_x.loc[user_number].astype(np.int64), x.astype(np.int64)),axis=1)


    # create a new data frame to store the top k similar users
    df_sim_users = df_sim.loc[df_sim.sort_values(by="distance").head(k).index]

    # calculate these similar users' rating on a given item, weighted by distance
    df_sim_users["weighed_d"] = map(lambda x: df_sim_users.loc[x]["distance"] * df_train_y.loc[x][item_number],
    df_sim_users.index)

    predicted = df_sim_users["weighed_d"].sum() / df_sim_users["distance"].sum()
    return predicted



# Remove books and users with less than 20 rating scores from the utility matrix by using for modifying the following codes.
def remove_users_items(dense_matrix,df_data1):
    dense_matrix = dense_matrix.fillna(0)
    dense_matrix.head()

    df_item_freq = df_data1.groupby("item").count()
    df_user_freq = df_data1.groupby("user").count()

    selected_items = df_item_freq[df_item_freq["rating"] > 20].index
    dense_matrix = dense_matrix[selected_items]
    selected_users = df_user_freq[df_user_freq["rating"] > 20].index
    dense_matrix = dense_matrix.loc[selected_users]

    return dense_matrix

def partition(dense_matrix):
    dense_matrix = dense_matrix.fillna(0)
    dense_matrix.head()

    df_x = dense_matrix[[col for col in dense_matrix.columns if col != 8010]]

    # create a series for the outcome
    #df_y = dense_matrix[['8010']]
    df_y = dense_matrix[[col for col in dense_matrix.columns if col != 8010]]

    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
    df_train_x = pd.DataFrame(train_x, columns=df_x.columns)
    df_test_x = pd.DataFrame(test_x, columns=df_x.columns)
    df_train_y = pd.DataFrame(train_y, columns=[8010])
    df_test_y = pd.DataFrame(test_y, columns=[8010])
    #print "shapes"
    #print df_train_x.shape
    #print df_test_x.shape
    #print df_train_y
    #print df_test_y
    print "Mean"
    df_train_x = df_train_x.fillna(0)
    df_test_x = df_test_x.fillna(0)
    df_train_y = df_train_y.fillna(0)
    df_test_y = df_test_y.fillna(0)
    print df_train_y[8010].astype(np.int64).mean()
    print df_test_y[8010].astype(np.int64).mean()

    return df_train_x, df_test_x, df_train_y,df_test_y


def main():
    # df_data1 = pd.read_csv("/Users/nikets/Downloads/INFS770_assignment3(1)/DBbook_train_ratings.tsv",
    df_data1 = pd.read_csv("/Users/aakankshasoral/Downloads/INFS770_assignment3/DBbook_train_ratings.tsv",
    # the location to the data file
    sep="\t", # for tab delimited documents, use "\t" as the seperator
    names=["user", "item", "rating"] # define the names for the four columns
    )

    df_data1.head()
    dense_matrix = df_data1.pivot_table(values="rating", index=["user"], columns=["item"], aggfunc=np.sum)
    uniqueData(dense_matrix,df_data1)
    correlationCheck(dense_matrix)

    # retrieve top five similar items to Item 8010
    topItems = top_k_items(dense_matrix,8010, 5)
    print topItems

    #remove_users_items(dense_matrix,df_data1)
    df_train_x,df_test_x,df_train_y,df_test_y = partition(dense_matrix)


    print
    user_number = df_test_x.index[23]
    item_number = 8010
    # This below line will throw error because User 24 is not available in df_test_y
    print ("Predicted rating on Book 8010 for 24th user:",
    user_based_predict(df_train_x, df_test_x, df_train_y, user_number, item_number, k=10))


    # This code is also throwing error
    pred_8010 = []
    for user_number in df_test_x.index:
        predicted = user_based_predict(df_train_x, df_test_x, df_train_y, user_number, 8010, k=10)
        pred_8010.append(predicted)

    # from sklearn.metrics import mean_absolute_error
    # print "Mean Absolute Error: ", mean_absolute_error(pred_8010, df_test_y['8010'])

    return


if __name__ == "__main__":
    main()
