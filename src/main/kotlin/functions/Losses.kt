package functions

import structure.Tensor
import kotlin.math.log

const val floatE = kotlin.math.E.toFloat()

fun crossEntropy(inTensors: ArrayList<Tensor>, labels: ArrayList<Tensor>): Float {
    var sum = 0F
    for ((i, tensor) in inTensors.withIndex()) {
        for ((iValue, value) in tensor.elements.withIndex()) {
            sum += log(value, floatE) * labels[i].elements[iValue]
        }
    }
    return -sum / inTensors.size
}

fun crossEntropyDer(inTensors: ArrayList<Tensor>, labels: ArrayList<Tensor>, outTensor: ArrayList<Tensor>) {
    for ((i, tensor) in inTensors.withIndex()) {
        for ((iValue, value) in tensor.elements.withIndex()) {
            outTensor[i].elements[iValue] = -(labels[i].elements[iValue] / value)
        }
    }
}

fun meanavg(inTensors: ArrayList<Tensor>, labels: ArrayList<Tensor>): Float {
    var sum = 0F
    for ((i, tensor) in inTensors.withIndex()) {
        for ((iValue, value) in tensor.elements.withIndex()) {
            sum += 0.5F * (value - labels[i].elements[iValue]) * (value - labels[i].elements[iValue])
        }
    }
    return sum
}

fun meanavgDer(inTensors: ArrayList<Tensor>, labels: ArrayList<Tensor>, outTensor: ArrayList<Tensor>) {
    for ((i, tensor) in inTensors.withIndex()) {
        for ((iValue, value) in tensor.elements.withIndex()) {
            outTensor[i].elements[iValue] = labels[i].elements[iValue] - value
        }
    }
}
