import functions.*
import inputs.InputMNIST
import inputs.InputSimple
import layers.*
import structure.Shape
import structure.Tensor

fun main() {
    val batchsize = 16
    val learningRate = 0.01F
    val epochs = 5

    //simpletest(batchsize, epochs, learningRate)
    mnistDense(batchsize, epochs, learningRate)
    //mnistConv(batchsize, epochs, learningRate)
}

fun mnistConv(batchsize: Int, epochs: Int, learningRate: Float) {
    val testNetworkConv = Network(
            Conv2D(Shape(5, 5, 1, 6), Shape(28, 28, 1), batchsize),
            Activation(::sigmoid, ::sigmoidDer),
            MaxPooling2D(Shape(24, 24 * 6), Shape(2, 2), batchsize),
            Conv2D(Shape(5, 5, 6, 16), Shape(12, 12, 6), batchsize),
            Activation(::sigmoid, ::sigmoidDer),
            MaxPooling2D(Shape(8, 8 * 16), Shape(2, 2), batchsize),
            FullyConnected(Shape(120, 4 * 4 * 16), batchsize),
            Activation(::sigmoid, ::sigmoidDer),
            FullyConnected(Shape(84, 120), batchsize),
            Activation(::sigmoid, ::sigmoidDer),
            FullyConnected(Shape(10, 84), batchsize),
            Softmax()
    )
    val train = InputMNIST("src/main/resources/MNIST_train.txt")
    val test = InputMNIST("src/main/resources/MNIST_test.txt")

    testNetworkConv.train(train, epochs, batchsize, learningRate, ::crossEntropy, ::crossEntropyDer, ::batchGradientDecent)
    testNetworkConv.evaluate(test)
}

fun mnistDense(batchsize: Int, epochs: Int, learningRate: Float) {
    val testNetworkDense = Network(
            FullyConnected(Shape(100, 784), batchsize),
            Activation(::sigmoid, ::sigmoidDer),
            FullyConnected(Shape(10, 100), batchsize),
            Softmax()
    )
    val train = InputMNIST("src/main/resources/MNIST_train.txt")
    val test = InputMNIST("src/main/resources/MNIST_test.txt")

    testNetworkDense.train(train, epochs, batchsize, learningRate, ::crossEntropy, ::crossEntropyDer, ::batchGradientDecent)
    testNetworkDense.evaluate(test)
}

fun simpletest(batchsize: Int, epochs: Int, learningRate: Float) {
    val tensor1 = Tensor(Shape(3, 3), false)
    tensor1.elements = floatArrayOf(-0.5057F, 0.3987F, -0.8943F, 0.3356F, 0.1673F, 0.8321F, -0.3485F, -0.4597F, -0.1121F)
    val tensor2 = Tensor(Shape(2, 3), false)
    tensor2.elements = floatArrayOf(0.4047F, 0.9563F, -0.8192F, -0.1274F, 0.3662F, -0.7252F)
    val label = Tensor(Shape(2, 1), false)
    label.elements = floatArrayOf(0.7095F, 0.0942F)

    val testNetwork = Network(
            FullyConnected(Shape(100, 784), batchsize),
            Activation(::sigmoid, ::sigmoidDer),
            FullyConnected(Shape(10, 100), batchsize),
            Softmax()
    )

    val inputDataTensors = listOf(Tensor(Shape(3, 1), false))
    inputDataTensors[0].elements[0] = 0.4183F
    inputDataTensors[0].elements[1] = 0.5209F
    inputDataTensors[0].elements[2] = 0.0291F
    val inputData = InputSimple(inputDataTensors, listOf(label))

    testNetwork.train(inputData, epochs, batchsize, learningRate, ::crossEntropy, ::crossEntropyDer, ::batchGradientDecent)
}

fun convtest(batchsize: Int, epochs: Int, learningRate: Float) {
    val tensor1 = Tensor(Shape(24, 1), false)
    tensor1.elements = floatArrayOf(0.1F, -0.2F, 0.5F, 0.6F, 1.2F, 1.4F, 1.6F, 2.2F, 0.01F, 0.2F, -0.3F, 4F, 0.9F, 0.3F, 0.5F, 0.65F, 1.1F, 0.7F, 2.2F, 4.4F, 3.2F, 1.7F, 6.3F, 8.2F)
    val tensor2 = Tensor(Shape(2, 2, 2), false)
    tensor2.elements = floatArrayOf(0.1F, -0.2F, 0.3F, 0.4F, 0.7F, 0.6F, 0.9F, -1.1F, 0.37F, -0.9F, 0.32F, 0.17F, 0.9F, 0.3F, 0.2F, -0.7F)
    val tensor3 = Tensor(Shape(2, 2, 2, 2), false)
    tensor3.elements = floatArrayOf(0.4F, 0.3F, -0.2F, 0.1F, 0.17F, 0.32F, -0.9F, 0.37F, -1.1F, 0.9F, 0.6F, 0.7F, -0.7F, 0.2F, 0.3F, 0.9F)
    val tensor4 = Tensor(Shape(3, 2, 2), false)
    tensor4.elements = floatArrayOf(0.1F, 0.33F, -0.6F, -0.25F, 1.3F, 0.01F, -0.5F, 0.2F, 0.1F, -0.8F, 0.81F, 1.1F)
    val tensor5 = Tensor(Shape(2, 2, 2, 2), false)

    val inputData = listOf(Tensor(Shape(3, 1), false))
    inputData[0].elements[0] = 0.4183F
    inputData[0].elements[1] = 0.5209F
    inputData[0].elements[2] = 0.0291F
    val inputLayer = InputSimple(listOf(tensor1), listOf(Tensor(Shape(1, 12), false)))

    val con2dlayer = Conv2D(Shape(2, 2, 2, 2), Shape(4, 3, 2), 1)
    con2dlayer.filter = tensor2
    val con2dlayer2 = Conv2D(Shape(2, 2, 2, 2), Shape(4, 3, 2), 1)
    con2dlayer2.filter = tensor3

    val testNetwork = Network(
            con2dlayer,
            Softmax(),
            con2dlayer2
    )
    testNetwork.train(inputLayer, epochs, batchsize, learningRate, ::crossEntropy, ::crossEntropyDer, ::batchGradientDecent)

    con2dlayer2.computedeltas(tensor4, tensor5, tensor1,tensor1)
    for (i in 0 until tensor5.shape.volume) {
        print(tensor5.elements[i])
        print(", ")
    }
}
