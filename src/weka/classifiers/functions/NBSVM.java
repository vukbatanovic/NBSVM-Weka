package weka.classifiers.functions;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.LibLINEAR_ModelModifier;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.SolverType;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.WekaException;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
  <!-- globalinfo-start -->
  * An NBSVM implementation capable of multiclass (one-vs-all) classification. It relies on the LibLINEAR library and its Weka wrapper class.<br><br>
  * Sida Wang, Christopher D. Manning: Baselines and Bigrams: Simple, Good Sentiment and Topic Classification, in Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012), pp. 90–94, Jeju Island, South Korea (2012). URL: nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
  *	<p/>
  <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;InProceedings{wang12simple,
 *    author = {Sida Wang and Christopher D. Manning},
 *    title = {Baselines and Bigrams: Simple, Good Sentiment and Topic Classification},
 *    booktitle = {Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012)},
 *    pages = {90-94},
 *    year = {2012},
 *    URL = {nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -S &lt;int&gt;
 *  Set type of solver (default: 1)
 *     0 -- L2-regularized logistic regression (primal)
 *     1 -- L2-regularized L2-loss support vector classification (dual)
 *     2 -- L2-regularized L2-loss support vector classification (primal)
 *     3 -- L2-regularized L1-loss support vector classification (dual)
 *     5 -- L1-regularized L2-loss support vector classification
 *     6 -- L1-regularized logistic regression
 *     7 -- L2-regularized logistic regression (dual)</pre>
 *
 * <pre> -C &lt;double&gt;
 *  Set the cost parameter C
 *   (default: 1)</pre>
 *
 * <pre> -Z
 *  Turn on normalization of input data (default: off)</pre>
 *
 * <pre>
 * -I &lt;int&gt;
 *  The maximum number of iterations to perform.
 *  (default 1000)
 * </pre>
 *
 * <pre> -P
 *  Use probability estimation (default: off)
 * currently for L2-regularized logistic regression only! </pre>
 *
 * <pre> -E &lt;double&gt;
 *  Set tolerance of termination criterion (default: 0.001)</pre>
 *
 * <pre> -W &lt;double&gt;
 *  Set the parameters C of class i to weight[i]*C
 *   (default: 1)</pre>
 *
 * <pre> -B &lt;double&gt;
 *  Add Bias term with the given value if &gt;= 0; if &lt; 0, no bias term added (default: 1)</pre>
 *
 * <pre> -A &lt;double&gt;
 *  Set the value of the Laplace smoothing parameter alpha (default: 1.0)</pre>
 *    
 * <pre> -X &lt;double&gt;
 *  Set the value of the interpolation parameter beta (default: 0.25)</pre>
 *
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *
 <!-- options-end -->
 *
 * @author Vuk Batanović
 * @see <i>Reliable Baselines for Sentiment Analysis in Resource-Limited Languages: The Serbian Movie Review Dataset</i>, Vuk Batanović, Boško Nikolić, Milan Milosavljević, in Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016), pp. 2688-2696, Portorož, Slovenia (2016).
 * <br>
 * https://github.com/vukbatanovic/NBSVM-Weka
 * <br>
 * @version 1.0.1
 */
public class NBSVM extends LibLINEAR {

	private static final long serialVersionUID = -884649357319242378L;
	
	public static final String REVISION = "1.0.1";

	/**
	 * The indexing value used to access the vector of r values in the r matrix when dealing with a two-class classification
	 */
	public static final int TWO_CLASSES_R_INDEX = 0;
	
    /** SVM solver types
     * <br>
     * Restated in this class in order to exclude the unsupported Crammer and Singer multiclassification and SV regression options from the
     * original LibLINEAR drop-down list
     * */
    public static final Tag[]      TAGS_SVMTYPE           = {new Tag(SolverType.L2R_LR.getId(), "L2-regularized logistic regression (primal)"),
            new Tag(SolverType.L2R_L2LOSS_SVC_DUAL.getId(), "L2-regularized L2-loss support vector classification (dual)"),
            new Tag(SolverType.L2R_L2LOSS_SVC.getId(), "L2-regularized L2-loss support vector classification (primal)"),
            new Tag(SolverType.L2R_L1LOSS_SVC_DUAL.getId(), "L2-regularized L1-loss support vector classification (dual)"),
            new Tag(SolverType.L1R_L2LOSS_SVC.getId(), "L1-regularized L2-loss support vector classification"),
            new Tag(SolverType.L1R_LR.getId(), "L1-regularized logistic regression"),
            new Tag(SolverType.L2R_LR_DUAL.getId(), "L2-regularized logistic regression (dual)")};
    
	/**
	 * The log-count ratio (r) values
	 */
	protected double [][] r;
	
	/**
	 * Computes the log-count ratio (r) values.
	 * <br>
	 * Portions of the code were taken from the NaiveBayesMultinomial class.
	 * <br>
	 * NaiveBayesMultinomial authors: Andrew Golightly (acg4@cs.waikato.ac.nz), Bernhard Pfahringer (bernhard@cs.waikato.ac.nz)
	 * @param instances The training data
	 */
	protected void initializeRValues(Instances instances) throws Exception {
		int numClasses = instances.numClasses();
	    int numAttributes = instances.numAttributes();
	    double [][] smoothedFeatureCountsGivenClass = new double[numClasses][numAttributes];
		r = new double [numClasses][numAttributes];
		
	    /*
	     * Initializing the matrix of feature counts
	     * NOTE: Laplace estimator is introduced in case a feature that does not appear for a class in the 
	     * training set does so for the test set
	    */
	    for(int c = 0; c < numClasses; c++) 
	    	for(int att = 0; att < numAttributes; att++)
	    		smoothedFeatureCountsGivenClass[c][att] = m_alpha;
		
	    //enumerate through the instances 
	    Instance instance;
	    double[] featuresPerClass = new double[numClasses];
		
	    java.util.Enumeration<Instance> enumInsts = instances.enumerateInstances();
	    while (enumInsts.hasMoreElements()) {
			instance = (Instance) enumInsts.nextElement();
			int classIndex = (int)instance.value(instance.classIndex());
		    double numOccurences;
				
			for (int a = 0; a < instance.numValues(); a++)
				if ((instance.index(a) != instance.classIndex()) && (!instance.isMissingSparse(a))) {
					numOccurences = instance.valueSparse(a) * instance.weight();
					if(numOccurences < 0)
						throw new Exception("Numeric attribute values must all be greater or equal to zero.");
					featuresPerClass[classIndex] += numOccurences;
					smoothedFeatureCountsGivenClass[classIndex][instance.index(a)] += numOccurences;
				}
	    }
		
		// If there are two classes we use only one vector of r values and we store it in r[TWO_CLASSES_R_INDEX]
	    // In this case smoothedFeatureCountsGivenClass[0] contains the p value and smoothedFeatureCountsGivenClass[1] the q value (p, q - from the original paper)
		if (numClasses == 2) {
			for (int v=0; v<numAttributes; v++) {
				/*
		      	 *	Normalizing smoothedFeatureCountsGivenClass values and saving each value as the log of each value
		      	 *	We reduce the denominator by one alpha since one attribute/feature is the class value
				 */
				for(int c = 0; c < numClasses; c++)
		    		smoothedFeatureCountsGivenClass[c][v] = Math.log(smoothedFeatureCountsGivenClass[c][v] / (featuresPerClass[c] + (numAttributes-1)*m_alpha));
				
				// log((p/|p|) / (q/|q|)) = log(p/|p|) - log(q/|q|)
				r[TWO_CLASSES_R_INDEX][v] = smoothedFeatureCountsGivenClass[0][v] - smoothedFeatureCountsGivenClass[1][v];
			}
		}
		/* 	If there are more than two classes we use a separate vector of r values for each class and we store each r vector in r[classIndex].
		 * 	Instead of using p for the smoothed feature counts in the positive class and q for the smoothed feature counts in the negative class, here we 
		 * 	just use p for the smoothed feature counts of the class currently considered in the one-vs-all setting. The not_p values are the smoothed
		 * 	feature counts for all the other classes taken together.
		 *	Class vectors of r values are calculated as log((p/|p|) / (not_p/|not_p|)) = log(p/|p|) - log(not_p/|not_p|)
		 */
		else {
			// The sum of all feature counts in all classes
			double featuresTotal = 0;
			// The sum of individual feature counts in all classes 
			double [] featureCountsSummedAcrossClasses = new double[numAttributes];
			
			for (int c = 0; c < numClasses; c++)
				featuresTotal += featuresPerClass[c];
			for (int v = 0; v < numAttributes; v++)
				for (int c = 0; c < numClasses; c++)
					// We subtract one alpha since we wish to include only the actual feature counts in the sum
					featureCountsSummedAcrossClasses[v] += (smoothedFeatureCountsGivenClass[c][v] - m_alpha);
			
			for(int v = 0; v < numAttributes; v++)
				for(int c = 0; c < numClasses; c++) {
					// The same calculation as in the two-class setting: log(p/|p|)
					r[c][v] = Math.log(smoothedFeatureCountsGivenClass[c][v] / (featuresPerClass[c] + (numAttributes-1)*m_alpha));
					
					/* Subtract log(not_p/|not_p|)
					 * We add two alpha values in the numerator since we wish to smooth the not_p "class group" as well 
					 * (one alpha is subtracted by subtracting the smoothedFeatureCountsGivenClass value)
					 */
					r[c][v] -= Math.log((featureCountsSummedAcrossClasses[v] - smoothedFeatureCountsGivenClass[c][v] + 2*m_alpha) / (featuresTotal - featuresPerClass[c] + (numAttributes-1)*m_alpha));
				}
		}
	}
		
	/**
	 * Returns the entire matrix of r values
	 */
	protected double [][] getR () {
		return r;
	}
		
	/**
	 * Returns the vector of r values for a particular class
	 */
	protected double [] getR (int classIndex) {
		return r[classIndex];
	}

	/**
	 * An auxiliary class used to apply NBSVM interpolation to the internal LibLINEAR model weights
	 */
	protected LibLINEAR_ModelModifier modelModifier = new LibLINEAR_ModelModifier ();
	
	/**
	 * Laplace smoothing parameter
	 */
	protected double m_alpha = 1.0;
	
	/**
	 * Interpolation parameter
	 */
	protected double m_beta = 0.25;
	
	/**
	 * An array of internal LibLINEAR models. In a two-class setting only models[TWO_CLASSES_R_INDEX] is used. In a multiclass setting each class has
	 * its own one-vs-all model.
	 */
	protected Model [] models;
	
	/**
	 * Sets the Laplace smoothing parameter
	 */
	public void setAlpha (double alpha) {
		this.m_alpha = alpha;
	}
	
	/**
	 * Returns the Laplace smoothing parameter
	 */
	public double getAlpha () {
		return m_alpha;
	}
	
	/**
	 * Sets the interpolation parameter beta
	 */
	public void setBeta (double beta) {
		this.m_beta = beta;
	}
	
	/**
	 * Returns the interpolation parameter beta
	 */
	public double getBeta () {
		return m_beta;
	}
	
	/**
     * Sets the type of SVM (default SVMTYPE_L2)
     *
     * @param value The type of the SVM
     */
	@Override
    public void setSVMType(SelectedTag value) {
        if (value.getTags() == TAGS_SVMTYPE) {
            setSolverType(SolverType.getById(value.getSelectedTag().getID()));
        }
    }
    
    /**
     * Gets the type of SVM
     *
     * @return The type of the SVM
     */
	@Override
    public SelectedTag getSVMType() {
        return new SelectedTag(m_SolverType.getId(), TAGS_SVMTYPE);
    }
	
	/**
     * Returns an instance turned into a sparse liblinear array.
     * <br>
     * Values are multiplied by the r values of the class given by classInd.
     * <br>
     * Most of the code for this method is taken from the LibLINEAR class. Original author: Benedikt Waldvogel (mail at bwaldvogel.de)
     * 
     * @param instance	The instance to work on
     * @param classInd	The index of the class whose one-vs-all model is being considered (or TWO_CLASSES_R_INDEX in a two-class setting)
     * @return		The liblinear array
     * @throws Exception	If setup of array fails
     */
    protected FeatureNode[] instanceToArray(Instance instance, int classInd) throws Exception {
        // vector of log-count ratio values for the class whose index is classInd
        double [] r = getR(classInd);
        
		///////////////////////////// Copied from LibLINEAR class /////////////////////////////////
    	// determine number of non-zero attributes
        int count = 0;
        
        for (int i = 0; i < instance.numValues(); i++) {
            if (instance.index(i) == instance.classIndex()) continue;
            if (instance.valueSparse(i) != 0) count++;
        }

        if (m_Bias >= 0) {
            count++;
        }

        // fill array
        FeatureNode[] nodes = new FeatureNode[count];
        int index = 0;
        for (int i = 0; i < instance.numValues(); i++) {

            int idx = instance.index(i);
            double val = instance.valueSparse(i);

            if (idx == instance.classIndex()) continue;
            if (val == 0) continue;
    	///////////////////////////////////////////////////////////////////////////////////////////
            
            // Multiply by the r value - Different from the original LibLINEAR method
            nodes[index] = new FeatureNode(idx + 1, val*r[idx]);
            
    	///////////////////////////// Copied from LibLINEAR class /////////////////////////////////
            index++;
        }

        // add bias term
        if (m_Bias >= 0) {
            nodes[index] = new FeatureNode(instance.numAttributes() + 1, m_Bias);
        }

        return nodes;
		///////////////////////////////////////////////////////////////////////////////////////////
    }

	/**
	 * Builds the classifier.
	 * <br>
	 * In the case of multiclass classification we build separate (one-vs-all) models for each class - we cannot use the built-in
	 * multiclass capability of LibLINEAR since we multiply the elements of feature vectors by different r values depending on the class
	 * that is being considered.
	 * <br>
	 * Portions of the code were taken from the LibLINEAR class. Original author: Benedikt Waldvogel (mail at bwaldvogel.de)
	 *
     * @param instances   The training data
     * @throws Exception  If liblinear classes not in classpath or liblinear
     *                    encounters a problem
     */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		///////////////////////////// Copied from LibLINEAR class /////////////////////////////////
		m_NominalToBinary = null;
	    m_Filter = null;
	    
	    getCapabilities().testWithFail(instances);

	    // remove instances with missing class
	    instances = new Instances(instances);
	    instances.deleteWithMissingClass();

		m_ReplaceMissingValues = new ReplaceMissingValues();
		m_ReplaceMissingValues.setInputFormat(instances);
		instances = Filter.useFilter(instances, m_ReplaceMissingValues);

		m_NominalToBinary = new NominalToBinary();
		m_NominalToBinary.setInputFormat(instances);
		instances = Filter.useFilter(instances, m_NominalToBinary);
		///////////////////////////////////////////////////////////////////////////////////////////
		
		initializeRValues(instances);
		int classNum = instances.numClasses();
		
		if (classNum == 2) {
			models = new Model[1];
			models[TWO_CLASSES_R_INDEX] = m_Model;
		}
		else {
			models = new Model[classNum];
			for (int i=0; i<classNum; i++)
				models[i] = new Model();
		}
		
		///////////////////////////// Copied from LibLINEAR class /////////////////////////////////
	    if (getNormalize()) {
	    	m_Filter = new Normalize();
	        m_Filter.setInputFormat(instances);
	        instances = Filter.useFilter(instances, m_Filter);
	    }
	    ///////////////////////////////////////////////////////////////////////////////////////////
	    
	    for (int modelInd = 0; modelInd < models.length; modelInd++) {
		    double[] vy = new double[instances.numInstances()];
		    FeatureNode[][] vx = new FeatureNode[instances.numInstances()][];
		    int max_index = 0;
	
		    for (int d = 0; d < instances.numInstances(); d++) {
		    	Instance inst = instances.instance(d);
		        FeatureNode[] x = instanceToArray(inst, modelInd);
		        if (x.length > 0) {
		        	max_index = Math.max(max_index, x[x.length - 1].index);
		        }
		        vx[d] = x;
		        double classValue = inst.classValue();
		        // Remapping the classes from a multiclass to multiple binary problems
		        if (classValue == modelInd)
		        	vy[d] = 1;
		        else
		        	vy[d] = -1;
		    }
	
		    ///////////////////////////// Copied from LibLINEAR class /////////////////////////////////
		    if (!m_Debug)
		        Linear.disableDebugOutput();
		    else
		        Linear.enableDebugOutput();
		        
		    // reset the PRNG for regression-stable results
		    Linear.resetRandom();
	
		    // train model
		    models[modelInd] = Linear.train(getProblem(vx, vy, max_index), getParameters());
		    ///////////////////////////////////////////////////////////////////////////////////////////

			// MNB / SVM interpolation - optional (set beta = 1 to turn off)
			double [] w = models[modelInd].getFeatureWeights();
			double mean = 0;
			for (int i = 0; i < w.length; i++)
				mean += Math.abs(w[i]);
			mean /= w.length;
			double [] interpolationW = new double [w.length];
			for (int i = 0; i < w.length; i++)
				interpolationW[i] = m_beta*w[i] + (1-m_beta)*mean;
			modelModifier.changeWeights(models[modelInd], interpolationW);
	    }
	}
	
	/**
     * Computes the distribution for a given instance.
     * <br>
     * Portions of the code were taken from the LibLINEAR class. Original author: Benedikt Waldvogel (mail at bwaldvogel.de)
     * 
     * @param instance The instance for which distribution is computed
     * @return The distribution
     * @throws Exception If the distribution can't be computed successfully
     */
	@Override
    public double[] distributionForInstance(Instance instance) throws Exception {
		///////////////////////////// Copied from LibLINEAR class /////////////////////////////////
		m_ReplaceMissingValues.input(instance);
		m_ReplaceMissingValues.batchFinished();
		instance = m_ReplaceMissingValues.output();

		m_NominalToBinary.input(instance);
		m_NominalToBinary.batchFinished();
		instance = m_NominalToBinary.output();

        if (m_Filter != null) {
            m_Filter.input(instance);
            m_Filter.batchFinished();
            instance = m_Filter.output();
        }
	    
        double[] result = new double[instance.numClasses()];
	    ///////////////////////////////////////////////////////////////////////////////////////////
        
        if (instance.classAttribute().isNominal() && (m_ProbabilityEstimates))
            if (m_SolverType != SolverType.L2R_LR && m_SolverType != SolverType.L2R_LR_DUAL && m_SolverType != SolverType.L1R_LR)
                throw new WekaException("Probability estimation is currently only " + "supported for logistic regression");
           
	    for (int modelInd = 0; modelInd < models.length; modelInd++) {
	        FeatureNode[] x = instanceToArray(instance, modelInd);
            double[] dec_values = new double[1];
            Linear.predictValues(models[modelInd], x, dec_values);
            // The result value is the distance from the separating hyperplane for the class that is being considered
			// If the distance is positive - the instance belongs to the class that is being considered; if it is negative - it does not
			// We do not remap the labels here since LibLINEAR always puts the +1 class at index 0, and we assigned the +1 value in training to the class whose binary one-vs-all classifier this is
            result[modelInd] = dec_values[0];
	    }
	    
	    if (!m_ProbabilityEstimates) {
	    	// In the multiclass setting, the chosen class is the one with the largest distance from the separating hyperplane
			// In a binary setting there is only one value - if it is greater than 0 (i.e. instance does belong to class[0]) then maxInd remains = 0, else it is changed to 1 
		    int maxInd = 0;
		    for (int i = 1; i < result.length; i++)
		    	if (result[i] > result[maxInd])
		    		maxInd = i;
		    
		    result = new double[instance.numClasses()];
		    result[maxInd] = 1;
	        return result;
	    }
	    else {
	    	// Calculates the probabilities in the same way as in the LibLINEAR and Linear classes
	    	double [] prob_estimates = new double[instance.numClasses()];
	    	for (int i = 0; i < prob_estimates.length; i++)
	            prob_estimates[i] = 1 / (1 + Math.exp(-result[i]));

	        if (instance.numClasses() == 2) // for binary classification
	            prob_estimates[1] = 1. - prob_estimates[0];
	        else {
	            double sum = 0;
	            for (int i = 0; i < instance.numClasses(); i++)
	                sum += prob_estimates[i];

	            for (int i = 0; i < instance.numClasses(); i++)
	                prob_estimates[i] = prob_estimates[i] / sum;
	        }
	        return prob_estimates;
	    }
    }
	
	
    /**
     * Returns a string describing classifier
     *
     * @return A description suitable for displaying in the
     *         explorer/experimenter gui
     */
	@Override
    public String globalInfo() {
        return "An NBSVM implementation capable of multiclass (one-vs-all) classification. It relies on the LibLINEAR library and its Weka wrapper class.\n\n" + getTechnicalInformation().toString();
    }
	
	/**
	 * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the original paper on NBSVM
	 */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Sida Wang, Christopher D. Manning");
        result.setValue(TechnicalInformation.Field.TITLE, "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification");
        result.setValue(TechnicalInformation.Field.YEAR, "2012");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012)");
        result.setValue(TechnicalInformation.Field.PAGES, "90-94");
        result.setValue(TechnicalInformation.Field.LOCATION, "Jeju Island, South Korea");
        result.setValue(TechnicalInformation.Field.URL, "http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf");

        return result;
    }
    
    /**
     * Returns an enumeration describing the available options.
     * Support vector regression and multiclass classification by Cramer and Singer are not supported
     * (the remaining options preserve the LibLINEAR numbering, for the sake of consistency).
     *
     * @return An enumeration of all the available options.
     */
    @Override
    @SuppressWarnings({ "rawtypes", "unchecked"})
    public Enumeration listOptions() {
    	
        Vector<Object> result = new Vector<Object>();
        result.addElement(new Option("\tSet type of solver (default: 1)\n" //
            + "\t\t 0 -- L2-regularized logistic regression (primal)\n" //
            + "\t\t 1 -- L2-regularized L2-loss support vector classification (dual)\n" //
            + "\t\t 2 -- L2-regularized L2-loss support vector classification (primal)\n" //
            + "\t\t 3 -- L2-regularized L1-loss support vector classification (dual)\n" //
            + "\t\t 5 -- L1-regularized L2-loss support vector classification\n" //
            + "\t\t 6 -- L1-regularized logistic regression\n" //
            + "\t\t 7 -- L2-regularized logistic regression (dual)\n", //
            "S", 1, "-S <int>"));

		///////////////////////////// Copied from LibLINEAR class /////////////////////////////////
        result.addElement(new Option("\tSet the cost parameter C\n" + "\t (default: 1)", "C", 1, "-C <double>"));

        result.addElement(new Option("\tTurn on normalization of input data (default: off)", "Z", 0, "-Z"));

        result.addElement(new Option("\tUse probability estimation (default: off)\n"
            + "currently for L2-regularized logistic regression, L1-regularized logistic regression or L2-regularized logistic regression (dual)! ", "P", 0,
            "-P"));

        result.addElement(new Option("\tSet tolerance of termination criterion (default: 0.001)", "E", 1, "-E <double>"));

        result.addElement(new Option("\tSet the parameters C of class i to weight[i]*C\n" + "\t (default: 1)", "W", 1, "-W <double>"));

        result.addElement(new Option("\tAdd Bias term with the given value if >= 0; if < 0, no bias term added (default: 1)", "B", 1, "-B <double>"));
	    
        result.addElement(new Option("\tThe maximum number of iterations to perform.\n" + "\t(default 1000)", "I", 1, "-I <int>"));
        ///////////////////////////////////////////////////////////////////////////////////////////
        
        
    	result.addElement(new Option("\tSet the Laplace smoothing parameter alpha\n" + "\t (default: 1.0)", "A", 1, "-A <double>"));
        
    	result.addElement(new Option("\tSet the interpolation parameter beta\n" + "\t (default: 0.25)", "X", 1, "-X <double>"));
        
    	
		//////////////////////// Copied from AbstractClassifier class /////////////////////////////
    	result.addElement(new Option(
        	      "\tIf set, classifier is run in debug mode and\n"
        	        + "\tmay output additional info to the console", "output-debug-info",
        	      0, "-output-debug-info"));
    	
    	result.addElement(new Option(
        	        "\tIf set, classifier capabilities are not checked before classifier is built\n"
        	          + "\t(use with caution).", "-do-not-check-capabilities", 0,
        	        "-do-not-check-capabilities"));
        result.addElement(new Option(
        	      "\tThe number of decimal places for the output of numbers in the model"
        	        + " (default " + m_numDecimalPlaces + ").",
        	      "num-decimal-places", 1, "-num-decimal-places"));
        result.addElement(new Option(
        	            "\tThe desired batch size for batch prediction " + " (default " + m_BatchSize + ").",
        	            "batch-size", 1, "-batch-size"));
	    ///////////////////////////////////////////////////////////////////////////////////////////
    	
        return result.elements();
    }
    
    /**
     * Sets the classifier options <p/>
     *
     <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -S &lt;int&gt;
     *  Set type of solver (default: 1)
     *   numbering is preserved from LibLINEAR
     *     0 -- L2-regularized logistic regression (primal)
     *     1 -- L2-regularized L2-loss support vector classification (dual)
     *     2 -- L2-regularized L2-loss support vector classification (primal)
     *     3 -- L2-regularized L1-loss support vector classification (dual)
     *     5 -- L1-regularized L2-loss support vector classification
     *     6 -- L1-regularized logistic regression
     *     7 -- L2-regularized logistic regression (dual)</pre>
     *
     * <pre> -C &lt;double&gt;
     *  Set the cost parameter C
     *   (default: 1)</pre>
     *
     * <pre> -Z
     *  Turn on normalization of input data (default: off)</pre>
     *
     * <pre>
     * -I &lt;int&gt;
     *  The maximum number of iterations to perform.
     *  (default 1000)
     * </pre>
     *
     * <pre> -P
     *  Use probability estimation (default: off)
     * currently for L2-regularized logistic regression only! </pre>
     *
     * <pre> -E &lt;double&gt;
     *  Set tolerance of termination criterion (default: 0.001)</pre>
     *
     * <pre> -W &lt;double&gt;
     *  Set the parameters C of class i to weight[i]*C
     *   (default: 1)</pre>
     *
     * <pre> -B &lt;double&gt;
     *  Add Bias term with the given value if &gt;= 0; if &lt; 0, no bias term added (default: 1)</pre>
     *  
     * <pre> -A &lt;double&gt;
     *  Set the value of the Laplace smoothing parameter alpha (default: 1.0)</pre>
     *    
     * <pre> -X &lt;double&gt;
     *  Set the value of the interpolation parameter beta (default: 0.25)</pre>
     *
     * <pre> -D
     *  If set, classifier is run in debug mode and
     *  may output additional info to the console</pre>
     *
     <!-- options-end -->
     *
     * @param options The options to parse
     * @throws Exception if parsing fails
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String tmpStr = Utils.getOption('A', options);
        if (tmpStr.length() != 0)
            setAlpha(Double.parseDouble(tmpStr));
        else
            setAlpha(1.0);
        
        tmpStr = Utils.getOption('X', options);
        if (tmpStr.length() != 0)
            setBeta(Double.parseDouble(tmpStr));
        else
            setBeta(0.25);

        super.setOptions(options);
    }
    
    /**
     * Returns the current options
     *
     * @return The current setup
     */
    @Override
    public String[] getOptions() {

        List<String> options = new ArrayList<String>();

        options.add("-S");
        options.add("" + m_SolverType.getId());

        options.add("-C");
        options.add("" + getCost());

        options.add("-E");
        options.add("" + getEps());

        options.add("-B");
        options.add("" + getBias());

        if (getNormalize()) options.add("-Z");

        if (getWeights().length() != 0) {
            options.add("-W");
            options.add("" + getWeights());
        }

        if (getProbabilityEstimates()) options.add("-P");
        
        options.add("-I");
        options.add("" + getMaximumNumberOfIterations());
        
        options.add("-A");
        options.add("" + getAlpha());
        
        options.add("-X");
        options.add("" + getBeta());

        return options.toArray(new String[options.size()]);
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return Tip text for this property suitable for
     *         displaying in the explorer/experimenter gui
     */
    public String alphaTipText() {
        return "The Laplace smoothing parameter.";
    }
 
    /**
     * Returns the tip text for this property
     *
     * @return Tip text for this property suitable for
     *         displaying in the explorer/experimenter gui
     */
    public String betaTipText() {
        return "The interpolation parameter.";
    }
    
    /**
     * Returns a string representation
     *
     * @return A string representation
     */
    @Override
    public String toString() {
        return "Weka's NBSVM implementation";
    }
    
    /**
     * Returns the revision string.
     *
     * @return The revision
     */
    @Override
    public String getRevision() {
        return REVISION;
    }
}