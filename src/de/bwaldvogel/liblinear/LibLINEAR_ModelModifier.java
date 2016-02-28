package de.bwaldvogel.liblinear;

import java.io.Serializable;

/**
 * An auxiliary class used to apply NBSVM interpolation to the internal LibLINEAR model weights
 * 
 * @author Vuk Batanović
 * @see "Reliable Baselines for Sentiment Analysis in Resource-Limited Languages: The Serbian Movie Review Dataset", Vuk Batanović, Boško Nikolić, Milan Milosavljević, in Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016), Portorož, Slovenia. (2016)
 * <br>
 * https://github.com/vukbatanovic/NBSVM-Weka
 * <br>
 * @version 1.0.0
 */
public class LibLINEAR_ModelModifier implements Serializable {

	private static final long serialVersionUID = -8046832126113372430L;

	/**
	 * Replaces the weight vector in the given model
	 */
	public void changeWeights (Model model, double [] weights) {
		model.w = Linear.copyOf(weights, weights.length);
	}
}
