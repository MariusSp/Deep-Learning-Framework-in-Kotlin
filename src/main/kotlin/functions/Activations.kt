package functions

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.tanh

fun sigmoid(x: Float): Float = 1F / (1F + exp(-x))

fun sigmoidDer(x: Float): Float = (1F / (1F + exp(-x))) * (1F - 1F / (1F + exp(-x)))

fun relu(x: Float): Float = max(0F, x)

fun reluDer(x: Float): Float = if (x <= 0F) 0F else 1F

fun tanH(x: Float): Float = tanh(x)

fun tanHDer(x: Float): Float = 1 - tanh(x).pow(2)
