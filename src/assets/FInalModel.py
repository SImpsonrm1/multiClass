import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Dense, Input, Normalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)




def main():
    file_path = 'FinalData.csv'
    data = pd.read_csv(file_path)
    
    # Setting the x and y values to the specific columns
    x = data.iloc[:, 1:-3]  # Features
    y = data.iloc[:, -3:]   # Target

    # Undoing the one hot encoding for the classifications
    y = y.idxmax(axis=1)
    map = {"Classification_Alex": 0, "Classification_Jordan": 1, "Classification_Ryan": 2}
    y = [map.get(item, item) for item in y]
    y = pd.DataFrame(y)
    
    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    np_data = data.to_numpy()
    print(np_data)
    y_data = np_data[:,-3:]
    print(len(y_data[0]))
    x_data = np_data[:,1:-3]
    print(x_data)

    # Keras Neural Network - UNCOMMENT THE BELOW LINE TO RE-TRAIN THE MODEL

    # clear_session()
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

    # norm_layer = Normalization(axis=1)
    # norm_layer.adapt(x_train)

    # nn_model = Sequential()
    # nn_model.add(Input(x_train.shape[1]))
    # nn_model.add(norm_layer)
    # nn_model.add(Dense(192, activation='relu', kernel_regularizer='l2'))
    # nn_model.add(Dropout(0.5))
    # nn_model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
    # nn_model.add(Dropout(0.5))
    # nn_model.add(Dense(16, activation='relu', kernel_regularizer='l2'))
    # nn_model.add(Dropout(0.5))
    # nn_model.add(Dense(3, activation='softmax'))
    # nn_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    # nn_model.summary()
    # history = nn_model.fit(x_train, y_train, epochs=2000, batch_size=4, verbose=1,
    #                     validation_data=[x_test, y_test], callbacks=[early_stopping])
    # nn_loss, nn_acc = nn_model.evaluate(x_test, y_test, verbose=0)
    # print(f"Neural Network Loss: {(nn_loss)}")
    # print(f"Neural Network Accuracy: {(nn_acc * 100):.2f}%")
    # nn_model.save('nn_model.keras')
    # nn_model.save('nn_model.h5')

    # print(f"Neural Network Training Accuracy: {(nn_acc * 100):.2f}%")

    # Keras Neural Network Training History
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['loss'])
    # plt.title('Neural Network Training History')
    # plt.ylabel('Accuracy/Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Accuracy', 'Loss'], loc='upper right')
    # plt.show()

    # Creating a Confusion Matrix for the test data
    x_valid = np.array(x_test)
    y_valid = np.array(y_test)

    from keras.models import load_model
    # Load saved model
    model = load_model('nn_model.h5', compile=True)
    model.summary()

    # Predict values from x_valid set
    prob_array = []
    for x in x_valid:
        prob_array.append(model.predict(np.array([x])))   # just loops through the x_valid set and predicts based off of each entry - be sure to cast the input to the model.predict to an np.array with an extra dimension

    y_pred = prob_to_predictions(prob_array) # prob_array is the 2D array of predictions

    conf_mat_dt = confusion_matrix(y_valid, y_pred)
    sns.heatmap(conf_mat_dt, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


def prob_to_predictions(array):
    # Use numpy.argmax to find the index of the max value in each row:
    max_indices = [np.argmax(row) for row in array]

    #turning the list of indices into a np array:
    predictions = np.array(max_indices)

    return predictions


if __name__ == '__main__':
    main()
