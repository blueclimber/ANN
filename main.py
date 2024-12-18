from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ann import one_hot_encoding, MLP
import matplotlib.pyplot as plt


def minibatch_generator(X, y, batch_size=100, random_seed=42):
    rng = np.random.RandomState(random_seed)
    indices = np.arange(X.shape[0])

    rng.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
        batch_idx = indices[start_idx: start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]


def compute_loss_acc(mlp, X, y, num_c, batch_size=100):
    mse = 0.
    correct_predict = 0
    num_examples = 0

    minibatch_gen = minibatch_generator(X, y, batch_size)

    for i, (x_features, classifications) in enumerate(minibatch_gen):

        # run a the classifier and get the results
        _, activation = mlp.forward(x_features)
        predict = np.argmax(activation, axis=1)
        one_hot_classifications = one_hot_encoding(classifications, num_c)

        # compile stats for calculating mse and accuracy
        loss = np.mean((one_hot_classifications - activation)**2)  
        correct_predict += (predict == classifications).sum()
        num_examples += classifications.shape[0]
        mse += loss

    # calculate mse and accuraccy 
    mse = mse/i
    acc = correct_predict/num_examples

    return mse, acc


def train(mlp, X_train, y_train, X_valid, y_valid, num_epochs, num_c, batch_size=100, learning_rate=0.1):
    epoch_loss = []
    epcoh_training_accuracy = []
    epoch_valid_accuracy = []

    for e in range(num_epochs):
        minibatch = minibatch_generator(X_train, y_train, batch_size)

        # run forward
        for X_mini, y_mini in minibatch:
            a_h, a_out = mlp.forward(X_mini)

            # calculate gradients by running backward
            loss_w_out, loss_b_out, loss_w_h, loss_b_h = mlp.backward(X_mini, y_mini, a_h, a_out)

            # update weights and bias
            mlp.w_out -= learning_rate * loss_w_out
            mlp.b_out -= learning_rate * loss_b_out
            mlp.w_h -= learning_rate * loss_w_h
            mlp.b_h -= learning_rate * loss_b_h

        # compute epoch MSE, accuraccy, and validation accuraccy
        train_loss, train_acc = compute_loss_acc(mlp, X_train, y_train, num_c, batch_size)
        _, valid_acc = compute_loss_acc(mlp, X_valid, y_valid, num_c, batch_size)

        epoch_loss.append(train_loss)
        epcoh_training_accuracy.append(train_acc)
        epoch_valid_accuracy.append(valid_acc)

        
        print(f"Epoch: {e+1:02d}/{num_epochs:02d} "
                f"| Training MSE: {train_loss:.02f} "
                f"| Training Accuracy: {train_acc:.02f} "
                f"| Validation Accuracy: {valid_acc:.02f}")
            
    return epoch_loss, epcoh_training_accuracy, epoch_valid_accuracy


def main():
    aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

    X = aids_clinical_trials_group_study_175.data.features


    y = aids_clinical_trials_group_study_175.data.targets

    X = X.to_numpy()
    y = y.to_numpy()

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=400, stratify=y)
    X_temp, X_valid, y_temp, y_valid = train_test_split(X_temp, y_temp, test_size=300, stratify=y_temp)
    X_train, _, y_train, _ = train_test_split(X_temp, y_temp, test_size=39, stratify=y_temp)

    sc = StandardScaler()
    sc.fit(X_train)

    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    X_valid = sc.transform(X_valid)

    mlp = MLP(num_f=X_train.shape[1], num_c=2, num_h=50)

    train_loss, train_acc, valid_acc = train(mlp, X_train, y_train, X_valid, y_valid, num_epochs=100, num_c=2)

    test_loss, test_acc = compute_loss_acc(mlp, X_test, y_test, num_c=2)

    print(f"Training MSE: {train_loss[-1]:.02f} | Testing MSE: {test_loss:.02f}")
    print(f"Training Accuracy: {train_acc[-1]:.02f} | Validation Accuracy: {valid_acc[-1]:.02f} | Testing Accuracy: {test_acc:.02f}")

    plt.plot(range(len(train_loss)), train_loss)
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Epoch")
    plt.show() 

    plt.plot(range(len(train_acc)), train_acc, label='Train Accuracy')
    plt.plot(range(len(valid_acc)), valid_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


