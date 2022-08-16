package layers

import structure.Shape
import structure.Tensor

class FullyConnected(shape: Shape, batchsize: Int) : ShapeChangingLayer {
    private var weights = Tensor(shape, true)
    private var indices = IntRange(0, batchsize - 1).toSet()
    private var bias = Tensor(Shape(shape.axis[0], 1), true)
    override val outputShape = Shape(weights.shape.axis[0], 1)
    private var biasUpdates = ArrayList(emptyList<Tensor>().toMutableList())
    private var derWeights = ArrayList(emptyList<Tensor>().toMutableList())

    init {
        for (i in 1..batchsize) {
            derWeights.add(Tensor(weights.shape, false))
        }
    }

    override fun forward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>) {
        if (outTensors.size != inTensors.size)
            throw Exception("Tensorlistshape mismatch")
        if (inTensors.first().elements.size > 100) {
            indices.parallelStream().parallel().forEach { weights.slowMatrixMul(inTensors[it], outTensors[it]) }
        } else {
            indices.parallelStream().parallel().forEach {
                weights.slowMatrixMul(inTensors[it], outTensors[it])
                for (i in 0 until bias.shape.axis[0]) {
                    outTensors[it].elements[i] += bias.elements[i]
                }
            }
        }
    }

    override fun backward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>, forwardInTensors: ArrayList<Tensor>) {
        if (outTensors.size != inTensors.size)
            throw Exception("Tensorlistshape mismatch")

        val save1 = Tensor(inTensors.first().shape.transpose(), false)
        val save2 = Tensor(forwardInTensors.first().shape.transpose(), false)
        indices.parallelStream().parallel().forEach { inTensors[it].slowMatrixMul(weights, outTensors[it]) }
        indices.parallelStream().parallel().forEach {
            inTensors[it].transpose(save1)
            forwardInTensors[it].transpose(save2)
            save1.slowMatrixMul(save2, derWeights[it])
        }
        biasUpdates = inTensors
    }

    override fun updateWeights(learningRate: Float, optimizer: (Float, ArrayList<Tensor>, Tensor) -> Unit) {
        optimizer(learningRate, derWeights, weights)
        optimizer(learningRate, biasUpdates, bias)

    }
}