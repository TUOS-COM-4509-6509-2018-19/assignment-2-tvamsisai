import pods
import zipfile
import sys
import pandas as pd
import numpy as np

def stocastic_objective_gradient(Y, U, V):
    learn_rate = 0.01
    nrows = Y.shape[0]
    obj = 0
    gU = pd.DataFrame(np.zeros((U.shape)), index=U.index)
    gV = pd.DataFrame(np.zeros((V.shape)), index=V.index)
    for i in range(1000):
        row = Y.iloc[np.random.randint(0, nrows)]
        user = row['users']
        film = row['movies']
        rating = row['ratings']
        prediction = np.dot(U.loc[user], V.loc[film]) # vTu
        diff = prediction - rating # vTu - y
        obj = diff*diff
        gU.loc[user] += 2*diff*V.loc[film]
        gV.loc[film] += 2*diff*U.loc[user]
        U -= learn_rate*gU
        V -= learn_rate*gV
    print(obj)
    return U, V, obj

ratings = pd.read_csv("./ml-latest-small/ratings.csv")
# YourStudentID = 460 # Include here the last three digits of your UCard number
indexes_unique_users = ratings['userId'].unique()
n_users = indexes_unique_users.shape[0]
nUsersInExample = n_users
# np.random.seed(YourStudentID)
indexes_users = np.random.permutation(n_users)
my_batch_users = indexes_users[0:nUsersInExample]
# We need to make a list of the movies that these users have watched
list_movies_each_user = [[] for _ in range(nUsersInExample)]
list_ratings_each_user = [[] for _ in range(nUsersInExample)]
# Movies
list_movies = ratings['movieId'][ratings['userId'] == my_batch_users[0]].values
list_movies_each_user[0] = list_movies                    
# Ratings                      
list_ratings = ratings['rating'][ratings['userId'] == my_batch_users[0]].values
list_ratings_each_user[0] = list_ratings
# Users
n_each_user = list_movies.shape[0]
list_users = my_batch_users[0]*np.ones((1, n_each_user))

for i in range(1, nUsersInExample):
    # Movies
    local_list_per_user_movies = ratings['movieId'][ratings['userId'] == my_batch_users[i]].values
    list_movies_each_user[i] = local_list_per_user_movies
    list_movies = np.append(list_movies, local_list_per_user_movies)
    # Ratings                                 
    local_list_per_user_ratings = ratings['rating'][ratings['userId'] == my_batch_users[i]].values
    list_ratings_each_user[i] = local_list_per_user_ratings
    list_ratings = np.append(list_ratings, local_list_per_user_ratings)  
    # Users                                   
    n_each_user = local_list_per_user_movies.shape[0]                                                                               
    local_rep_user =  my_batch_users[i]*np.ones((1, n_each_user))    
    list_users = np.append(list_users, local_rep_user)

# Let us first see how many unique movies have been rated
indexes_unique_movies = np.unique(list_movies)
n_movies = indexes_unique_movies.shape[0]
# As it is expected no all users have rated all movies. We will build a matrix Y 
# with NaN inputs and fill according to the data for each user 
temp = np.empty((n_movies,nUsersInExample,))
temp[:] = np.nan
Y_with_NaNs = pd.DataFrame(temp)
for i in range(nUsersInExample):
    local_movies = list_movies_each_user[i]
    ixs = np.in1d(indexes_unique_movies, local_movies)
    Y_with_NaNs.loc[ixs, i] = list_ratings_each_user[i]
Y_with_NaNs.index = indexes_unique_movies.tolist()
Y_with_NaNs.columns = my_batch_users.tolist()

p_list_ratings = np.concatenate(list_ratings_each_user).ravel()
p_list_ratings_original = p_list_ratings.tolist()
mean_ratings_train = np.mean(p_list_ratings)
p_list_ratings =  p_list_ratings - mean_ratings_train # remove the mean
p_list_movies = np.concatenate(list_movies_each_user).ravel().tolist()
p_list_users = list_users.tolist()
Y = pd.DataFrame({'users': p_list_users, 'movies': p_list_movies, 'ratingsorig': p_list_ratings_original,'ratings':p_list_ratings.tolist()})

q = 2
U = pd.DataFrame(np.random.normal(size=(nUsersInExample, q))*0.0001, index=my_batch_users)
V = pd.DataFrame(np.random.normal(size=(n_movies, q))*0.0001, index=indexes_unique_movies)
obj = 0.
#while True:
for i in range(50):
    U, V, obj = stocastic_objective_gradient(Y, U, V)
#     display.clear_output()
#     plt.plot(V.iloc1:, 0], V.iloc[:, 1], 'bx')
#    print(obj)
#    if obj < 0.00000001:
#        break
# plt.plot(V.iloc[:, 0], V.iloc[:, 1], 'bx')

def get_prediction(Y, U, V):
    ratings = []
    pred_ratings = []
    error_ratings = []
    for i in range(Y.shape[0]):
        ratings.append(Y.iloc[i]['ratings'])
        pred_ratings.append(np.dot(U.loc[Y.iloc[i]['users']], V.loc[Y.iloc[i]['movies']]))
        error_ratings.append(abs(ratings[-1] - pred_ratings[-1]))
    Y['error'] = error_ratings
    Y['pred_ratings'] = pred_ratings + mean_ratings_train
    return ratings, pred_ratings, error_ratings

_, _, error = get_prediction(Y, U, V)