#Imports

import os
import warnings
import time
import sqlite3
import pandas as pd
import hdf5_getters
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

#End Imports

#Global Variables:

#A list of all ADDITIONAL attribute names present in an h5 file.
attributeNames = ['num_songs', 'artist_playmeid', 'artist_7digitalid', 'artist_latitude',
    'artist_longitude', 'artist_location', 'similar_artists', 'artist_terms',
    'artist_terms_freq', 'artist_terms_weight', 'analysis_sample_rate', 'audio_md5',
    'danceability', 'end_of_fade_in', 'energy', 'key', 'key_confidence', 'loudness',
    'mode', 'mode_confidence', 'start_of_fade_out', 'tempo', 'time_signature',
    'time_signature_confidence', 'segments_start', 'segments_confidence', 'segments_pitches',
    'segments_timbre', 'segments_loudness_max', 'segments_loudness_max_time',
    'segments_loudness_start', 'sections_start', 'sections_confidence', 'beats_start',
    'beats_confidence', 'bars_start', 'bars_confidence', 'tatums_start', 'tatums_confidence',
    'artist_mbtags', 'artist_mbtags_count']

#A list of best found ADDITIONAL attribute names found from exploratory feature analysis in model.ipynb
#After feature analysis has been performed, it is recommended to only append necessary attributes, as appendAdditionalAttributes()
#is very slow to run.
bestAttributes = ['end_of_fade_in', 'key', 'loudness', 'mode', 'start_of_fade_out', 'tempo', 'time_signature']

#A list of attributes included from track_metadata.db that are not relavent to modeling.
#This list exists to remove attributes when not adding all attributes (from variable `attributeNames`) initially.
startingAttributesToRemove = ['track_id', 'song_id', 'artist_id', 'artist_mbid', 'track_7digitalid', 'shs_perf', 'shs_work']

#Specifies attribute names that represent irrelavent IDs.
idColumns = ['track_id', 'song_id', 'artist_id', 'artist_mbid', 'track_7digitalid', 'artist_playmeid', 'artist_7digitalid']

#Specifies attributes to be dropped (and reasoning) for our model implementation.
uselessAttributes = [ 'num_songs', #Always 1
    'shs_perf', #Unknown attribute, 637 instances of -1
    'shs_work', #Unknown attribute, 616 instances of 0
    'artist_latitude', #499 instances of NaN
    'artist_longitude', #499 instances of NaN
    'artist_location', #Redundant if including latitude and longitude
    'similar_artists', #Mapping would be too complex. List of artist IDs.

    'artist_terms', #Contains genres. Could be added back as categorical attribute.
    'artist_terms_freq', #Frequencies for artist_terms. These should be used together.
    'artist_terms_weight', #Weights for artist_terms. These should be used together.

    'analysis_sample_rate', #Always has value of 22050
    'audio_md5', #Checksum value that we do not need for modeling.
    'danceability', #Always 0
    'energy', #Always 0
    'key_confidence', #Confidence score for the musical key a song is set in. Not useful because we don't have confidence scores for other possible keys.
    'mode_confidence', #This could be more useful than key_confidence, because mode is binary (0 or 1), and we know the confidence of the inverse. Might add this back.
    'time_signature_confidence', #Confidence score for time signature of a song. No reference to other confidence score for other possible time signatures.

    #All of these could be useful, but not sure yet how to handle arrays for model.ipynb
    'segments_start', 'segments_confidence', 'segments_pitches', 'segments_timbre',	'segments_loudness_max', 'segments_loudness_max_time', 'segments_loudness_start',
    'sections_start', 'sections_confidence', 'beats_start', 'beats_confidence', 'bars_start', 'bars_confidence', 'tatums_start', 'tatums_confidence',

    'artist_mbtags', #More labels, stands for 'musicbrainz tags'. Hard to deal with.
    'artist_mbtags_count' #Unique counts for artist_mbtags.
    ]

#Measure duration to run each function
programStartTime = time.time()

#End Global Variables

#Constants
N_COMPONENTS = 2
RANDOM_STATE = 42
#End Constants

#Note: The first user is specified with 'user1', and the second user with 'user2'.

def main():
    #Filter warnings
    warnings.filterwarnings('ignore')

    #Load in all songs from Million Song Dataset into DataFrame.
    print("Loading songs in from Million Song Dataset to DataFrame...")
    startTime = time.time()
    df = songsToDataframe('track_metadata.db')
    executionTime(startTime)

    #Create Spotify Listening History specific DataFrames for user1 and user2.
    print("Creating user DataFrames matching Spotify listening history...")
    startTime = time.time()
    userDF1, userDF2 = getUserSpotifyDataframe(df)
    executionTime(startTime)

    #Add columns with additional attributes to each user specific DataFrame.
    #Note: Both of these functions take over a minute to run each.
    print("Appending additional song attributes to user DataFrames...")
    startTime = time.time()
    userDF1 = appendAdditionalAttributes(userDF1, 'user1', bestAttributes)
    userDF2 = appendAdditionalAttributes(userDF2, 'user2', bestAttributes)
    executionTime(startTime)

    #Combine both user DataFrames column-wise, adding a `user` column specifying each row with 'user1' or 'user2' respectively.
    print("Combining both user DataFrames...")
    startTime = time.time()
    combinedDF = combineDataframes(userDF1, userDF2, 'user1', 'user2')
    executionTime(startTime)

    #Remove unnecessary initial columns from the dataset.
    print("Removing specified columns from combined DataFrame...")
    startTime = time.time()
    filteredDF = filterAttributes(combinedDF, startingAttributesToRemove)
    executionTime(startTime)

    #Clean the Data to prepare for modelling.
    print("Cleaning data in combined DataFrame...")
    startTime = time.time()
    cleanedDF = cleanData(filteredDF)
    executionTime(startTime)

    #Setup variable X to be modelled on (remove additional label columns).
    print("Defining input DataFrame X for model...")
    startTime = time.time()
    X = filterAttributes(cleanedDF, ['title', 'release', 'artist_name', 'user'])
    executionTime(startTime)

    #Get a dictionary of all information related to the model.
    print("Training model with data: X...")
    startTime = time.time()
    modelData = trainModel(X, N_COMPONENTS, RANDOM_STATE)
    executionTime(startTime) 

    #Append the silhouette sample score of each song respectively to the cleaned DataFrame
    print("Appending silhouette sample score column to cleaned DataFrame...")
    startTime = time.time()
    scoreDF = appendSilhouetteSamples(modelData['silhouetteSamples'], cleanedDF)
    executionTime(startTime)

    #Create individual DataFrames from the clusters. The number of DataFrames is equal to the number of components.
    print("Creating DataFrames representing each cluster (mixture)...")
    startTime = time.time()
    clusteredSongsDataframes = clustersToDataframes(N_COMPONENTS, modelData['labels'], scoreDF, order = True)
    executionTime(startTime)

    #Export the cluster DataFrames to individual .csv files.
    print("Exporting clustered DataFrames to .csv files...")
    startTime = time.time()
    dataframesToCsv(clusteredSongsDataframes)
    executionTime(startTime)

def songsToDataframe(databaseFileName):
    """ Loads in all the contents of Million Song Dataset to a
        Pandas Dataframe.

    Parameters:
    databaseFileName (str): Name of database file.

    Returns:
    df (DataFrame): Pandas DataFrame containing all of the content.
    """

    connection = sqlite3.connect(databaseFileName)
    df = pd.read_sql_query("SELECT * FROM songs", connection)
    connection.close()

    return df

def getUserSpotifyDataframe(allDataDF):
    """ Filters the Million Song Dataset to match 2 users Spotify
        Listening history as per .h5 files, from the directory `h5Files/user1_h5Files`
        and from `h5Files/user2_h5Files`

    Parameters:
    allDataDF (DataFrame): A Pandas DataFrame containing all of the Million
        Song Dataset data.

    Returns:
    user1DF (DataFrame): Pandas DataFrame filtered to match a user1's Spotify
        Listening history.
    user2DF (DataFrame): Pandas DataFrame filtered to match a user2's Spotify
        Listening history.
    """

    # #Read in list of .h5 file names from a text file.
    user1_h5FileNames = os.listdir(os.path.join('.', 'h5Files', 'user1_h5Files'))
    user2_h5FileNames = os.listdir(os.path.join('.', 'h5Files', 'user2_h5Files'))

    #Remove the .h5 file extension from each file name.
    user1_h5FileNames = [fileName.replace('.h5', '') for fileName in user1_h5FileNames]
    user2_h5FileNames = [fileName.replace('.h5', '') for fileName in user2_h5FileNames]
    #Filter allDataDF based on matching songs.
    user1DF = allDataDF.loc[allDataDF['track_id'].isin(user1_h5FileNames)]
    user2DF = allDataDF.loc[allDataDF['track_id'].isin(user2_h5FileNames)]

    return user1DF, user2DF

def getAttribute(track_id, function, user):
    """ Get additional hdf5 song attributes
        given a track_id.
        Note: This is currently hard-coded to match specific users.

        Parameters:
        track_id (str): Unique identifier for a given song.
            Part of filename.
        function (func): hdf5 function to call.
        user (str): Used to specify 'Ben' vs. 'Katie'.

        Returns:
        value (dynamic): Returned value from function.
    """
    
    if user == 'user1':
        filePath = 'h5Files\\user1_h5Files\\' + track_id + '.h5'
    elif user == 'user2':
        filePath = 'h5Files\\user2_h5Files\\' + track_id + '.h5'
    
    h5 = hdf5_getters.open_h5_file_read(filePath)
    value = function(h5)

    h5.close()
    return value

def appendAdditionalAttributes(userDF, user, attributeNames):
    """ Get additional song features from an HDF5 song file (same name as `track_id`)
        and append them as additional columns for each song in userDF, based off of
        attributes defined in the list `attributeNames`.

    Parameters:
    userDF (DataFrame): The user specific DataFrame to append additional features to.
    user (str): A string representing a specific user. Used to define file location of h5 files.
    attributeNames: List of attributes to add as additional features for the model.
        A list of attribute names is defined in global variables.

    Returns:
    userDF (DataFrame): Updated DataFrame with additional attributes.
    """

    for attribute in attributeNames:
        #Double check that the syntax of `attribute` (which is a variable) in the line below.
        userDF[attribute] = userDF['track_id'].apply(getAttribute, args = (getattr(hdf5_getters, 'get_' + attribute), user))

    return userDF

def combineDataframes(userDF1, userDF2, user1, user2):
    """ Combine 2 Pandas DataFrames into 1 with a new reference
        column specifying where each row of data originated from.
        DataFrames are combined column-wise.

    Parameters:
    userDF1 (DataFrame): The first DataFrame to be combined.
    userDF2 (DataFrame): The second DataFrame to be combined.
    user1 (str): The name of the first user, to appear as a label in each row.
    user2 (str): The name of the second user, to appear as a label in each row.

    Returns:
    combinedDF (DataFrame): The column-wise combined DataFrame.
    """
    #Reset index before merging dataframes.
    userDF1 = userDF1.reset_index(drop = True)
    userDF2 = userDF2.reset_index(drop = True)

    #Add label to each row before merging.
    userDF1['user'] = user1
    userDF2['user'] = user2

    combinedDF = pd.concat([userDF1, userDF2])
    combinedDF = combinedDF.reset_index(drop = True)

    return combinedDF

def filterAttributes(df, attributes):
    """ Remove columns from DataFrame specified by attributesList.

    Parameters:
    df (DataFrame): The Pandas DataFrame to remove attributes from.
    attributes (list): A list of attribute names matching column names in df.
    
    Returns:
    df (DataFrame): The filtered Pandas DataFrame.
    """

    df = df.drop(columns = attributes)

    return df

def cleanData(df):
    """ Define/Modify additional feature attributes, normalize the data,
        and impute missing values. If a different set of features are used to train
        the model, this function must be modified.

    Parameters:
    df (DataFrame): The Pandas DataFrame to be cleaned.
    
    Returns:
    df (DataFrame): The cleaned Pandas DataFrame.
    """

    #`end_of_fade_in` and `start_of_fade_out` can be normalized from 0 to 1 based off of `duration`.
    df['end_of_fade_in'] = df['end_of_fade_in'] / df['duration']
    df['start_of_fade_out'] = df['start_of_fade_out'] / df['duration']

    #Replace missing year values and impute them as mean year, scaling afterwards (will then become attribute defining how recent each song is, instead of year being categorical.)
    df['year'].replace(0, np.nan, inplace = True)
    meanImputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    df['year'] = meanImputer.fit_transform(df[['year']])
    df['year'] = df['year'].round().astype(int)

    #Min Max Scale remaining each remaining attribute
    minMaxScaler = MinMaxScaler()
    attributesToScale = ['duration', 'loudness', 'tempo', 'time_signature', 'year', 'key']
    for attribute in attributesToScale:
        df[attribute] = minMaxScaler.fit_transform(df[[attribute]])

    return df

def trainModel(X, n_components, random_state):
    """ Train a GaussianMixture model based off of DataFrame X,
        and return all of the models attributes as a dictionary.

    Parameters:
    X (DataFrame): Input data for the model.
    n_components (int): Number of mixture components (clusters) the model will form.
    random_state (int): Sets a random state to ensure the model reproduces the same results.

    Returns:
    modelData (dictionary): A dictionary containing all of the models attributes.

    """
    gmModel = GaussianMixture(n_components = n_components, random_state = random_state)
    gmModel.fit(X)

    #The mean values for each mixture component (n dimensional array of n components, k features).
    meanValues = gmModel.means_

    #Get the prediction label (cluser grouping) for each assigned mixture.
    labels = gmModel.predict(X)

    #Get the prediction probability for each assigned label.
    #Note: For 2 components, each prediction will be 100%. We could select n_components > 2, and reassign labels based
    #      off of prediction probabilities to ensure the same number of data points for each individual label (equal weights).
    #      Example: From 500 rows in X, and n_components = 5: Suppose 150 data points were assigned with label `0`.
    #      Take the 100 highest probability data points and keep them for label `0`, and reassign the remaining 50
    #      based off of their second highest prediction probability.
    probabilities = gmModel.predict_proba(X)

    #Get the number of iterations performed by the best fit of Expectation-Maximization before reaching convergence.
    nIterations = gmModel.n_iter_

    #Get the precision matrix (inverse of covariance matrix) for each component in the mixture.
    componentPrecisions = gmModel.precisions_

    #Get the Bayesian Information Criteria score for data 'X' in the model.
    bicScore = gmModel.bic(X)

    #Get the number of features the model is trained on.
    nFeatures = gmModel.n_features_in_

    #Get the weights for each mixture component (percentage of data points for each cluster.)
    weights = gmModel.weights_

    #Get the silhouette score for the model based on the input data X, their predictions (`labels`) and n_components.
    silhouetteAverage = silhouette_score(X, labels)

    #Get the silhouette samples for each data point (how similar a sample is to it's own cluster, compared to other clusters).
    #Provides insight to how well related a song is to it's grouping.
    silhouetteSamples = silhouette_samples(X, labels)

    modelData = {
        'gmModel': gmModel,
        'meanValues': meanValues,
        'labels': labels,
        'probabilities': probabilities,
        'nIterations': nIterations,
        'componentPrecisions': componentPrecisions,
        'bicScore': bicScore,
        'nFeatures': nFeatures,
        'weights': weights,
        'silhouetteAverage': silhouetteAverage,
        'silhouetteSamples': silhouetteSamples
    }

    return modelData

def clustersToDataframes(n_components, labels, df, order = False):
    """ Create n_components number of DataFrames, corresponding to the clustering
        of songs as defined by the labels of the GMM.

    Parameters:
    n_components (int): The number of clustering mixture components for the GMM.
    labels (list): A list of integers representing which cluster each data point (song)
        belongs to.
    df (DataFrame): The input (user combined) DataFrame to reference song indices from.
    order (bool): If set to true, orders the DataFrames based off of silhouette_samples scoring.
        Default is false. Call appendSilhouetteSamples() to add scoring prior to setting this to true.

    returns:
    dfs (list): A list of DataFrame objects, corresponding to each label. The length
        of this list is equal to the number of clusters (or n_components, #unique labels).
    """
    mappings = [] #2D list of len(n_components) representing index mapping for each label. 
    dfs = [] #List holding DataFrames to return
    sortedDFs = [] #List holding sorted DataFrames to return when order = True

    for i in range(n_components):
        temp = []
        for j in range(len(labels)):
            if labels[j] == i:
                temp.append(j)
        mappings.append(temp)

    for i in range(n_components):
        newDF = df.iloc[mappings[i]]
        newDF = newDF.filter(items = ['title', 'release', 'artist_name', 'user', 'silhouette_score'])
        dfs.append(newDF)

    if order:
        for df in dfs:
            sortedDFs.append(df.sort_values(by = 'silhouette_score', ascending = False))
        return sortedDFs

    return dfs

def assignNewSongToCluster(gmModel, X):
    """ Given a pre-trained Gaussian Mixture Model, assign new songs (X) to defined
        clusters (predict labels for data X).
    
    Parameters:
    gmModel (GaussianMixture): The object representing the pre-trained GMM.
    X (DataFrame): DataFrame of song values to be assigned labels (given clusters).

    Returns:
    labels (list): A list containing the labels assigned to each song in X.
        Same length as that of X.
    """

    return gmModel.predict(X)

def dataframesToCsv(dfs, fileNames = []):
    """ Output each DataFrame in the list dfs to it's own csv file. By default,
        the naming convention is 'cluster#.csv' where # represent an int incrementing from 0.
        Optionally, you can define a list of file names for the csv files.
        Note: Currently no error checking for incorrect fileNames formatting.
    
    Parameters:
    dfs (list): A list of DataFrames that will each be exported to a .csv file.
    fileNames (list): A list of csv file names (with .csv extension) used for naming export files.
        Example: ["outputA.csv", "outputB.csv"]

    Returns:
    none
    """
    #Case where function called without list of fileNames
    if not fileNames:
        #Export the cluster DataFrames to individual .csv files.
        count = 0 #Used to label csv files
        for df in dfs:
            df.to_csv('cluster' + str(count) + '.csv')
            count += 1
    #Case where list of fileNames provided
    else:
        for i in range(len(dfs)):
            dfs[i].to_csv(fileNames[i])

def executionTime(startTime):
    """ Used to print run times for each function, and elapsed program running time.
    
    Parameters:
    startTime(float): The starting time before a function call.

    Returns:
    none
    """
    endTime = time.time()

    functionDuration = str(round(endTime - startTime, 4))
    programDuration = str(round(endTime - programStartTime, 4))

    print("Function Duration:\t", functionDuration + 's', "\nElapsed Time:\t\t", programDuration + 's', "\n")

def appendSilhouetteSamples(silhouetteSamples, df):
    """ Append each silhouette_sample to combined DataFrame.
        Represents alikeness of each song to cluster center.

    Parameters:
    silhouetteSamples (list): List of floats representing the silhouette score
        for each data point.
    df (DataFrame): The DataFrame to add a `silhouette_score` column to.
    
    Returns:
    df (DataFrame): A DataFrame with the appended scoring column.
    """
    df['silhouette_score'] = silhouetteSamples

    return df

if __name__ == "__main__":
    main()