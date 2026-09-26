import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)



c0 = -3
c1 = -8


X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
], dtype=float)


Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)



class MLP:

    def __init__(self):

        np.random.seed(1)

        self.W1 = np.random.uniform(
            -1,
            1,
            (2, 2)
        )

        self.W2 = np.random.uniform(
            -1,
            1,
            (2, 1)
        )


    def train(self,
              X,
              Y,
              lr=0.5,
              max_epochs=20000,
              error_limit=0.001):


        for epoch in range(max_epochs):

            total_error = 0


            for i in range(len(X)):


                x = X[i].reshape(1, 2)

                y = Y[i].reshape(1, 1)



                hidden_input = x @ self.W1

                hidden_output = sigmoid(hidden_input)



                final_input = hidden_output @ self.W2

                output = sigmoid(final_input)



                error = y - output


                total_error += float(error[0][0] ** 2)



                output_delta = (
                    error *
                    sigmoid_derivative(output)
                )


                hidden_delta = (
                    output_delta @ self.W2.T *
                    sigmoid_derivative(hidden_output)
                )



                self.W2 += (
                    hidden_output.T @ output_delta
                    *
                    lr
                )


                self.W1 += (
                    x.T @ hidden_delta
                    *
                    lr
                )



            if epoch % 1000 == 0:

                print(
                    f"MLP эпоха {epoch}, ошибка {total_error:.6f}"
                )



            if total_error <= error_limit:

                return epoch + 1, total_error



        return max_epochs, total_error



    def predict(self, x):

        hidden = sigmoid(
            x @ self.W1
        )

        output = sigmoid(
            hidden @ self.W2
        )

        return output





class SinglePerceptron:


    def __init__(self):

        np.random.seed(1)

        self.W = np.random.uniform(
            -1,
            1,
            (2,1)
        )



    def train(self,
              X,
              Y,
              lr=0.5,
              max_epochs=10000):


        for epoch in range(max_epochs):

            error_sum = 0


            for i in range(len(X)):


                x = X[i].reshape(1,2)

                y = Y[i].reshape(1,1)



                output = sigmoid(
                    x @ self.W
                )



                error = y - output


                error_sum += float(error[0][0] ** 2)



                delta = (
                    error *
                    sigmoid_derivative(output)
                )



                self.W += (
                    x.T *
                    delta *
                    lr
                )



            if epoch % 1000 == 0:

                print(
                    f"Однослойный эпоха {epoch}, ошибка {error_sum:.6f}"
                )



        return max_epochs, error_sum



    def predict(self,x):

        return sigmoid(
            x @ self.W
        )





def accuracy(model):

    correct = 0


    for i in range(len(X)):


        result = model.predict(
            X[i].reshape(1,2)
        )


        prediction = 1 if result[0][0] >= 0.5 else 0


        if prediction == Y[i][0]:

            correct += 1



    return correct / len(X) * 100





mlp = MLP()


mlp_epochs, mlp_error = mlp.train(
    X,
    Y
)



single = SinglePerceptron()


single_epochs, single_error = single.train(
    X,
    Y
)




print("\n==============================")
print("РЕЗУЛЬТАТЫ")
print("==============================")


print(
    f"""
Модель                 Эпохи       Ошибка       Accuracy

MLP 2-2-1              {mlp_epochs:<10} {mlp_error:.6f}     {accuracy(mlp):.2f}%

Однослойный            {single_epochs:<10} {single_error:.6f}     {accuracy(single):.2f}%
"""
)




def convert_input(a,b):


    if a <= -5.5:
        a = -1
    else:
        a = 1


    if b <= -5.5:
        b = -1
    else:
        b = 1


    return np.array([[a,b]])





print("\n==============================")
print("ПРОВЕРКА СЕТИ")
print("==============================")


while True:

    try:

        A = float(
            input("Введите A (-10...10): ")
        )

        B = float(
            input("Введите B (-10...10): ")
        )


        if not (-10 <= A <= 10 and -10 <= B <= 10):

            print("Числа должны быть от -10 до 10")

            continue



        data = convert_input(A,B)


        result = mlp.predict(data)


        y = float(result[0][0])


        print(
            "Выход сети ŷ =",
            round(y,5)
        )


        if abs(y-0) < abs(y-1):

            print(
                "Ближайший класс: c0 = -3"
            )

        else:

            print(
                "Ближайший класс: c1 = -8"
            )



    except:

        print("Работа завершена")

        break