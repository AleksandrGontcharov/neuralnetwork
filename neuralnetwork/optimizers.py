# Optimizers
import numpy as np

# if you want to use no optimizer, then use momentum with beta = 0


def momentum():
    def optimizer(dW, dB, **kwargs):
        """ MOMENTUM OPTIMIZER - with bias correction
        Given dB, dW of the current mini batch
        and v_dW, v_dB of the previous gradient update,
        returns an updated v_dW, v_dB
        """
        beta = kwargs["beta"]

        V_dW = kwargs["V_dW"]
        V_dB = kwargs["V_dB"]

        V_dW = beta * V_dW + (1 - beta) * dW
        V_dB = beta * V_dB + (1 - beta) * dB

        _ = ""

        return V_dW, V_dB, _, _

    def weight_update(bias_correction=True, **kwargs):
        """ Rule on how to update the weights using this optimizer
        with bias correction!
        """
        learning_rate = kwargs["learning_rate"]
        V_d = kwargs["V_d"]
        beta = kwargs["beta"]
        iteration = kwargs["iteration"]

        if bias_correction:
            correction = 1 - beta ** iteration
        else:
            correction = 1

        update =  learning_rate * (V_d) / correction

        return update

    return optimizer, weight_update


def RMSprop():
    def optimizer(dW, dB, **kwargs):
        """ Given dB, dW of the current mini batch
        and v_dW, v_dB of the previous gradient update,
        returns an updated v_dW, v_dB
        """
        V_dW = kwargs["V_dW"]
        V_dB = kwargs["V_dB"]
        beta = kwargs["beta"]

        V_dW = beta * V_dW + (1 - beta) * np.square(dW)
        V_dB = beta * V_dB + (1 - beta) * np.square(dB)

        _ = ""

        return V_dW, V_dB, _, _

    def weight_update(**kwargs):
        """ Rule on how to update the weights using this optimizer
        """
        learning_rate = kwargs["learning_rate"]
        V_d = kwargs["V_d"]
        gradient = kwargs["gradient"]  # either dW or dB

        update = learning_rate * np.divide(gradient, np.sqrt(V_d) + 1e-08)

        return update

    return optimizer, weight_update


def nesterov_momentum():
    """ Not sure if implementation is correct
    """

    def optimizer(dW, dB, **kwargs):
        """ NESTEROV MOMENTUM OPTIMIZER
        Given dB, dW of the current mini batch
        and v_dW, v_dB of the previous gradient update,
        returns an updated v_dW, v_dB
        """
        V_dW = kwargs["V_dW"]
        V_dB = kwargs["V_dB"]
        beta = kwargs["beta"]
        learning_rate = kwargs["learning_rate"]
        network = kwargs["network"]
        X = kwargs["X"]
        Y = kwargs["Y"]
        key = kwargs["key"]

        # Update weights in the network to a temporary W_look_ahead

        network.layers[key]["weights"] = (
            network.layers[key]["weights"] - learning_rate * beta * V_dW
        )
        network.layers[key]["biases"] = (
            network.layers[key]["biases"] - learning_rate * beta * V_dB
        )

        # Get gradients for this temporary network configuration
        grads = network.backward(X, Y)
        dW_ahead = grads[key]["dW"]
        dB_ahead = grads[key]["dB"]

        V_dW = beta * V_dW + (1 - beta) * dW_ahead
        V_dB = beta * V_dB + (1 - beta) * dB_ahead

        _ = ""

        return V_dW, V_dB, _, _

    def weight_update(**kwargs):
        """ Rule on how to update the weights using this optimizer
        """
        learning_rate = kwargs["learning_rate"]
        V_d = kwargs["V_d"]

        update = learning_rate * V_d

        return update

    return optimizer, weight_update


def adam():
    def optimizer(dW, dB, **kwargs):
        """ Adam Optimizer
        """

        V_dW = kwargs["V_dW"]
        V_dB = kwargs["V_dB"]
        S_dW = kwargs["S_dW"]
        S_dB = kwargs["S_dB"]
        beta1 = kwargs["beta"]
        beta2 = kwargs["beta2"]

        # Momentum like terms
        V_dW = beta1 * V_dW + (1 - beta1) * dW
        V_dB = beta1 * V_dB + (1 - beta1) * dB

        # RMSprop like terms
        S_dW = beta2 * S_dW + (1 - beta2) * np.square(dW)
        S_dB = beta2 * S_dB + (1 - beta2) * np.square(dB)

        return V_dW, V_dB, S_dW, S_dB

    def weight_update(**kwargs):
        """ Rule on how to update the weights using this optimizer
        """
        learning_rate = kwargs["learning_rate"]
        V_d = kwargs["V_d"]
        S_d = kwargs["S_d"]

        # terms for bias correction
        beta1 = kwargs["beta"]
        beta2 = kwargs["beta2"]
        iteration = kwargs["iteration"]

        V_d = V_d / (1 - beta1 ** iteration)
        S_d = S_d / (1 - beta2 ** iteration)

        update =  learning_rate * np.divide(V_d, np.sqrt(S_d) + 1e-08)

        return update

    return optimizer, weight_update
