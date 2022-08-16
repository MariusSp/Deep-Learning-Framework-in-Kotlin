package layers

import structure.Shape
import structure.Tensor
import kotlin.math.exp

class Softmax : Layer {
    override fun forward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>) {
        if (inTensors.size != outTensors.size) {
            throw Exception("listsize mismatch")
        }
        inTensors.indices.toSet().parallelStream().parallel().forEach { activationFunction(inTensors[it], outTensors[it]) }
    }

    override fun backward(inTensors: ArrayList<Tensor>, outTensors: ArrayList<Tensor>, forwardInTensors: ArrayList<Tensor>) {
        inTensors.indices.toSet().parallelStream().parallel().forEach { deractivationFunction(inTensors[it], outTensors[it], forwardInTensors[it]) }
    }

    private fun deractivationFunction(inTensor: Tensor, outTensor: Tensor, forwardInTensor: Tensor) {
        val max = forwardInTensor.elements.max() ?: throw Exception("no maximum")
        forwardInTensor.elements = forwardInTensor.elements.map { it - max }.toFloatArray()
        forwardInTensor.elements = forwardInTensor.elements.map { exp(it) }.toFloatArray()
        val sum = forwardInTensor.elements.sum()
        val derTensor = Tensor(Shape(forwardInTensor.shape.axis[0], forwardInTensor.shape.axis[0]), false)
        for (i in 0 until forwardInTensor.shape.axis[0]) {
            for (j in 0 until forwardInTensor.shape.axis[0]) {
                // ite spalte jte zeile
                if (i == j) {
                    derTensor.elements[j + i * forwardInTensor.shape.axis[0]] = forwardInTensor.elements[i] * (sum - forwardInTensor.elements[i]) / (sum * sum)
                } else {
                    derTensor.elements[j + i * forwardInTensor.shape.axis[0]] = -forwardInTensor.elements[i] * forwardInTensor.elements[j] / (sum * sum)
                }
            }
        }
        inTensor.slowMatrixMul(derTensor, outTensor)
    }

    private fun activationFunction(inTensor: Tensor, outTensor: Tensor) {
        val max = inTensor.elements.max() ?: throw Exception("no maximum")
        outTensor.elements = inTensor.elements.map { it - max }.toFloatArray()
        outTensor.elements = outTensor.elements.map { exp(it) }.toFloatArray()
        val sum = outTensor.elements.sum()
        outTensor.elements = outTensor.elements.map { it / sum }.toFloatArray()
    }
}