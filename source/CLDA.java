package cc.mallet.topics;

import java.util.Arrays;
import java.util.HashMap;
import java.io.*;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;

@Deprecated
public class CLDA implements Serializable {

	int numDocs;
	int numtrainDocs;
	double[][] phi;
	int numTopics; // Number of topics to be fit
	double alpha;  // Dirichlet(alpha,alpha,...) is the distribution over topics
	double beta;   // Prior on per-topic multinomial distribution over words
	double tAlpha;
	double vBeta;
	InstanceList ilist;  // the data field of the instances is expected to hold a FeatureSequence
	int[][] topics; // indexed by <document index, sequence index>
	int numTypes=4819;
	
	int numTrainTokens;
	int numTestTokens;
	int numWords;
	int[][] docTopicCounts; // indexed by <document index, topic index>
	int[][] typeTopicCounts; // indexed by <feature index, topic index>
	int[] tokensPerTopic; // indexed by <topic index>
	
	int[][] concepts; // indexed by <document index, sequence index>
	int[][] probabilityWordConcept;
	int[][] numProbabilityWordConcept;
	int[] perWordConceptNumber;
	String[] vocabulary;
	int conceptInd=0;
	
	int[] nonConceptWords;
	int numofNunConceptWords=0;
	String[] vocConcepts;
	String[] truevocConcepts;
	int[] conceptWords;
	int numofConceptWords=0;
	
	int[] prowcsum;
	int[] sumc;
	
	private HashMap<String,Integer> conceptMap = new HashMap<String, Integer>();

	public CLDA (int numberOfTopics)
	{
		this (numberOfTopics, 50.0, 0.01);
	}

	public CLDA (int numberOfTopics, double alphaSum, double beta)
	{
		this.numTopics = numberOfTopics;
		this.alpha = alphaSum / numTopics;
		this.beta = beta;
	}

	public void estimate (InstanceList documents, int numIterations, int showTopicsInterval,
                     int outputModelInterval, String outputModelFilename,
                     Randoms r) throws IOException
	{
		ilist = documents.shallowClone();
		numWords = ilist.getDataAlphabet().size ();
		numDocs = ilist.size();
		numtrainDocs=(int) (numDocs*0.8);
		topics = new int[numDocs][];
		docTopicCounts = new int[numDocs][numTopics];
		tokensPerTopic = new int[numTopics];
		vocabulary= new String[numWords];
		prowcsum = new int[numWords];
		
		probabilityWordConcept = new int[numWords][];
		numProbabilityWordConcept = new int[numWords][];
		numofNunConceptWords=0;
		nonConceptWords = new int[numWords];
		conceptWords = new int[numWords];
		getProbability();
		numTypes=conceptMap.size();
		truevocConcepts = new String[numTypes];
		sumc = new int[numTypes];
		for(String c : conceptMap.keySet()){
        	int tmpCnt = conceptMap.get(c);
        	truevocConcepts[tmpCnt]=vocConcepts[Integer.valueOf(c).intValue()];
        }
		phi = new double[numTopics][numWords];
		System.out.println("concept="+numTypes);
		typeTopicCounts = new int[numTypes+numofNunConceptWords][numTopics];		
		tAlpha = alpha * numTopics;
		vBeta = beta * (numTypes+numofNunConceptWords);
		concepts = new int[numDocs][]; // indexed by <document index, sequence index>
		
		Arrays.fill(prowcsum, 0);
		for (int i = 0; i < numWords; i++) {
			if(nonConceptWords[i]==-1)
			{
				for (int conceptInd = 0; conceptInd < probabilityWordConcept[i].length; conceptInd++) 
				{
					prowcsum[i] += probabilityWordConcept[i][conceptInd];
				}
			}
		}
		
		Arrays.fill(sumc, 0);
		for (int i = 0; i < numWords; i++) {
			if(nonConceptWords[i]==-1)
			{
				for (int conceptInd = 0; conceptInd < probabilityWordConcept[i].length ; conceptInd++) 
				{
					sumc[numProbabilityWordConcept[i][conceptInd]] += probabilityWordConcept[i][conceptInd];
				}
			}
		}
		int topic, seqLen, concept;
		FeatureSequence fs;
		for (int di = 0; di < numDocs; di++) {
			try {
				fs = (FeatureSequence) ilist.get(di).getData();
			} catch (ClassCastException e) {
				System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
                         +"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
				throw e;
			}
			seqLen = fs.getLength();
			if(di<numtrainDocs)
				numTrainTokens += seqLen;
			else
				numTestTokens += seqLen;
			topics[di] = new int[seqLen];
			concepts[di] = new int[seqLen];
			// Randomly assign tokens to topics
			for (int si = 0; si < seqLen; si++) {
				int wordId=fs.getIndexAtPosition(si);
				if(nonConceptWords[wordId]==-1)
				{
					concept = numProbabilityWordConcept[wordId][r.nextInt(probabilityWordConcept[wordId].length)];
					concepts[di][si]=concept;
					topic = r.nextInt(numTopics);
					topics[di][si] = topic;
					docTopicCounts[di][topic]++;
					//typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
					typeTopicCounts[concept][topic]++;
					tokensPerTopic[topic]++;
				}
				else {
					topic = r.nextInt(numTopics);
					topics[di][si] = topic;
					docTopicCounts[di][topic]++;
					//typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
					typeTopicCounts[numTypes+nonConceptWords[wordId]][topic]++;
					tokensPerTopic[topic]++;
				}
				
			}
		}
 
		this.estimate(0, numtrainDocs, numIterations, showTopicsInterval, outputModelInterval, outputModelFilename, r);
		// 124.5 seconds
		// 144.8 seconds after using FeatureSequence instead of tokens[][] array
		// 121.6 seconds after putting "final" on FeatureSequence.getIndexAtPosition()
		// 106.3 seconds after avoiding array lookup in inner loop with a temporary variable

	}
	
	public void initializeTestVarible (Randoms r)
	{
		int topic, seqLen, concept;
		FeatureSequence fs;
		for (int di = numtrainDocs; di < numDocs; di++) {
			try {
				fs = (FeatureSequence) ilist.get(di).getData();
			} catch (ClassCastException e) {
				System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
						+"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
				throw e;
			}
			seqLen = fs.getLength();
			Arrays.fill(docTopicCounts[di], 0);
			// Randomly assign tokens to topics
			for (int si = 0; si < seqLen; si++) {
				int wordId=fs.getIndexAtPosition(si);
				if(nonConceptWords[wordId]==-1)
				{
					concept = numProbabilityWordConcept[wordId][r.nextInt(probabilityWordConcept[wordId].length)];
					concepts[di][si]=concept;
					topic = r.nextInt(numTopics);
					topics[di][si] = topic;
					docTopicCounts[di][topic]++;
				}
				else {
					topic = r.nextInt(numTopics);
					topics[di][si] = topic;
					docTopicCounts[di][topic]++;
				}
		
			}
		}
	}

	
	public void getProbability() throws IOException {
		vocConcepts= new String[numTypes];

		String vocString = null;	    
		Arrays.fill(nonConceptWords, -1);
		int numConcepts=0;
		
		File vocConceptFile = new File("data/vocconcept.txt");
		BufferedReader vocConceptReader = new BufferedReader(new FileReader(vocConceptFile));
		String tmpvocConcept = null;
		while ((tmpvocConcept = vocConceptReader.readLine()) != null){
			vocConcepts[numConcepts++]=tmpvocConcept;
		}
		vocConceptReader.close();
		//vocString=(String) ilist.getDataAlphabet().lookupObject(voc);
		File proWordConceptFile = new File("data/ap_relation");
		BufferedReader proWordConceptReader =  new BufferedReader(new FileReader(proWordConceptFile));
		String tempString = null;
		int conceptnum=0;
		while ((tempString = proWordConceptReader.readLine()) != null){
			String[] temp=tempString.split("\t");
			int wordInd = Integer.valueOf(temp[0]).intValue();
			if(temp.length==1)
			{
				nonConceptWords[wordInd]=numofNunConceptWords;
				numofNunConceptWords++;
			}
			else {
				numProbabilityWordConcept[wordInd]=new int[temp.length-1];
				probabilityWordConcept[wordInd]=new int[temp.length-1];
				for (int i = 1; i < temp.length; i++) {
					String[] pro=temp[i].split(" ");
					if(!conceptMap.containsKey(pro[0]))
					{
						conceptMap.put(pro[0], conceptnum++);
					}
					numProbabilityWordConcept[wordInd][i-1]=conceptMap.get(pro[0]);
					probabilityWordConcept[wordInd][i-1]=Integer.valueOf(pro[1]).intValue();
				}
			}
		}
		proWordConceptReader.close();
	}

	/* Perform several rounds of Gibbs sampling on the documents in the given range. */ 
	public void estimate (int docIndexStart, int docIndexLength,
	                      int numIterations, int showTopicsInterval,
                     int outputModelInterval, String outputModelFilename,
                     Randoms r) throws IOException
	{
		File perplexityFile = new File("data/perplexity_clda"+numTopics); // this is the file for labels
		Writer perplexityWriter = new OutputStreamWriter(new FileOutputStream(perplexityFile));
		perplexityWriter.write("words number = "+numWords+"\n");
		long startTime = System.currentTimeMillis();
		for (int iterations = 0; iterations < numIterations; iterations++) {
			if (iterations % 10 == 0) System.out.print (iterations);	else System.out.print (".");
			System.out.flush();
			sampleTopicsForDocs(0, numtrainDocs-1, r);
			if (showTopicsInterval != 0 && iterations % showTopicsInterval == 0 && iterations > 0) {
				
				System.out.println ();
//				logger.info("<" + iteration + "> Log Likelihood: " + modelLogLikelihood() + "\n" +
//						topWords (wordsPerTopic));
				this.write (new File(outputModelFilename+'.'+iterations));
				double trainperplexityValue=calPerplexity(0,numtrainDocs,numTrainTokens);
				initializeTestVarible(r);
				for (int i = 0; i < numIterations; i++) {
					sampleTopicsForTestDocs(numtrainDocs, numDocs-numtrainDocs, r);
				}
				double testperplexityValue=calPerplexity(numtrainDocs,numDocs,numTestTokens);
				long nowTime = System.currentTimeMillis();
				System.out.println("<" + iterations + "> train perplexity: " + trainperplexityValue + "\n");
				System.out.println("<" + iterations + "> test perplexity: " + testperplexityValue + "\n");
				perplexityWriter.write("<" + iterations + "> train perplexity: " + trainperplexityValue + "\nTime : "+ (nowTime-startTime) );
				perplexityWriter.write("<" + iterations + "> test perplexity: " + testperplexityValue);
				//printTopWords (5, false);
			}
			
		}

		long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
		long minutes = seconds / 60;	seconds %= 60;
		long hours = minutes / 60;	minutes %= 60;
		long days = hours / 24;	hours %= 24;
		System.out.print ("\nTotal time: ");
		if (days != 0) { System.out.print(days); System.out.print(" days "); }
		if (hours != 0) { System.out.print(hours); System.out.print(" hours "); }
		if (minutes != 0) { System.out.print(minutes); System.out.print(" minutes "); }
		System.out.print(seconds); System.out.println(" seconds");
		perplexityWriter.close();
	}

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsForAllDocs (Randoms r)
	{
		//double[] topicWeights = new double[numTopics];
		// Loop over every word in the corpus
		for (int di = 0; di < numtrainDocs; di++) {
			sampleTopicsForOneDoc ((FeatureSequence)ilist.get(di).getData(),
			                       topics[di], docTopicCounts[di], r , di);
		}
	}

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsForDocs (int start, int length, Randoms r)
	{
		assert (start+length <= docTopicCounts.length);
		//double[] topicWeights = new double[numTopics*numTypes];
		// Loop over every word in the corpus
		for (int di = start; di < start+length; di++) {
			sampleTopicsForOneDoc ((FeatureSequence)ilist.get(di).getData(),
			                       topics[di], docTopicCounts[di], r,di);
		}
	}

private void sampleTopicsForOneDoc (FeatureSequence oneDocTokens, int[] oneDocTopics, // indexed by seq position
	                                    int[] oneDocTopicCounts, // indexed by topic index
	                                     Randoms r ,int di)
	{
		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLen = oneDocTokens.getLength();
		double tw;
		// Iterate over the positions (words) in the document
		for (int si = 0; si < docLen; si++) {
			//type = oneDocTokens.getIndexAtPosition(si);
			//type = oneDocConcept[si];
			oldTopic = oneDocTopics[si];
			// Remove this token from all counts
			oneDocTopicCounts[oldTopic]--;
			//typeTopicCounts[type][oldTopic]--;
			tokensPerTopic[oldTopic]--;
			// Build a distribution over topics for this token
			
			topicWeightsSum = 0;
			
			int wordId=oneDocTokens.getIndexAtPosition(si);
			
			if(nonConceptWords[wordId]==-1)
			{
				type = concepts[di][si];
				double[] topicWeights = new double[numTopics];
				Arrays.fill (topicWeights, 0.0);
				
				typeTopicCounts[type][oldTopic]--;
				
				for (int ti = 0; ti < numTopics; ti++) {
					tw = ((typeTopicCounts[type][ti] + beta) / (tokensPerTopic[ti] + vBeta))
					      * ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
					topicWeightsSum += tw;
					topicWeights[ti] = tw;
				}
				newTopic = r.nextDiscrete (topicWeights, topicWeightsSum);
				oneDocTopics[si] = newTopic;
				topics[di][si] = newTopic;
				oneDocTopicCounts[newTopic]++;
				tokensPerTopic[newTopic]++;
				double typeWeightsSum = 0;
				double[] typeWeights = new double[probabilityWordConcept[wordId].length];
				Arrays.fill (typeWeights, 0.0);
				for (int cpt = 0; cpt < probabilityWordConcept[wordId].length; cpt++) {
					tw = 1.0*probabilityWordConcept[wordId][cpt]/sumc[numProbabilityWordConcept[wordId][cpt]]*(typeTopicCounts[numProbabilityWordConcept[wordId][cpt]][newTopic] + beta);
					typeWeightsSum += tw;
					typeWeights[cpt] = tw;
				}
				// Sample a topic assignment from this distribution
				type=numProbabilityWordConcept[wordId][r.nextDiscrete (typeWeights, typeWeightsSum)];
				concepts[di][si]=type;
				typeTopicCounts[type][newTopic]++;
			}
			else {
				type = numTypes+nonConceptWords[wordId];
				double[] topicWeights = new double[numTopics];
				Arrays.fill (topicWeights, 0.0);
				
				typeTopicCounts[type][oldTopic]--;
				
				currentTypeTopicCounts = typeTopicCounts[type];
				for (int ti = 0; ti < numTopics; ti++) {
					tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
						      * ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
					topicWeightsSum += tw;
					topicWeights[ti] = tw;
				}
				newTopic = r.nextDiscrete (topicWeights, topicWeightsSum);

				oneDocTopics[si] = newTopic;
				oneDocTopicCounts[newTopic]++;
				typeTopicCounts[type][newTopic]++;
				tokensPerTopic[newTopic]++;
			}
		}
	}
	
	public void sampleTopicsForTestDocs (int start, int length, Randoms r)
	{
		assert (start+length <= docTopicCounts.length);
		for (int di = start; di < start+length; di++) {
			sampleTopicsForTest ((FeatureSequence)ilist.get(di).getData(),
		                       topics[di], docTopicCounts[di], r,di);
		}
	}

	private void sampleTopicsForTest (FeatureSequence oneDocTokens, int[] oneDocTopics, // indexed by seq position
			int[] oneDocTopicCounts, // indexed by topic index
			Randoms r ,int di)
	{
		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLen = oneDocTokens.getLength();
		double tw;
		// Iterate over the positions (words) in the document
		for (int si = 0; si < docLen; si++) {
			oldTopic = oneDocTopics[si];
			oneDocTopicCounts[oldTopic]--;

			topicWeightsSum = 0;

			int wordId=oneDocTokens.getIndexAtPosition(si);

			if(nonConceptWords[wordId]==-1)
			{
				type = concepts[di][si];
				double[] topicWeights = new double[numTopics];
				Arrays.fill (topicWeights, 0.0);
				for (int ti = 0; ti < numTopics; ti++) {
					tw = ((typeTopicCounts[type][ti] + beta) / (tokensPerTopic[ti] + vBeta))
							* ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
					topicWeightsSum += tw;
					topicWeights[ti] = tw;
				}
				newTopic = r.nextDiscrete (topicWeights, topicWeightsSum);
				oneDocTopics[si] = newTopic;
				topics[di][si] = newTopic;
				oneDocTopicCounts[newTopic]++;
				double typeWeightsSum = 0;
				double[] typeWeights = new double[probabilityWordConcept[wordId].length];
				Arrays.fill (typeWeights, 0.0);
				for (int cpt = 0; cpt < probabilityWordConcept[wordId].length; cpt++) {
					tw = 1.0*probabilityWordConcept[wordId][cpt]/sumc[numProbabilityWordConcept[wordId][cpt]]*(typeTopicCounts[numProbabilityWordConcept[wordId][cpt]][newTopic] + beta);
					typeWeightsSum += tw;
					typeWeights[cpt] = tw;
				}
				// Sample a topic assignment from this distribution
				type=numProbabilityWordConcept[wordId][r.nextDiscrete (typeWeights, typeWeightsSum)];
				concepts[di][si]=type;
			}
			else {
				type = numTypes+nonConceptWords[wordId];
				double[] topicWeights = new double[numTopics];
				Arrays.fill (topicWeights, 0.0);
				currentTypeTopicCounts = typeTopicCounts[type];
				for (int ti = 0; ti < numTopics; ti++) {
					tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
							* ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
					topicWeightsSum += tw;
					topicWeights[ti] = tw;
				}
				// Sample a topic assignment from this distribution
				newTopic = r.nextDiscrete (topicWeights, topicWeightsSum);

				// Put that new topic into the counts
				oneDocTopics[si] = newTopic;
				oneDocTopicCounts[newTopic]++;
			}
		}
	}

	public double calPerplexity(int docStartInd, int docEndInd,int numTokens){
		//System.out.println("Perplexity computation starts...");
		double logPerplexity = 0.0;
		int[] oneDocTopicCnt;
		double[] topicProbabilities;
		double logTwo = Math.log(2.0);
		topicProbabilities=new double[numTopics];
		FeatureSequence fs;
		for (int docInd = docStartInd; docInd <docEndInd ; docInd++){
			try {
				fs = (FeatureSequence) ilist.get(docInd).getData();
			} catch (ClassCastException e) {
				System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
                         +"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
				throw e;
			}
			oneDocTopicCnt = docTopicCounts[docInd];
			//considering single word therefore hardcode group size of 1
			for (int topicInd=0; topicInd < numTopics; topicInd++){
				topicProbabilities[topicInd] =(alpha + oneDocTopicCnt[topicInd])
										/(tAlpha + fs.getLength());				
			}
		
			for (int wi=0;wi<fs.getLength();wi++){
				int word=fs.getIndexAtPosition(wi);
				double total = 0.0;
				if(nonConceptWords[word]==-1)
				{
						
					for(int topicInd= 0; topicInd< numTopics; topicInd++) {
						for (int conceptInd = 0; conceptInd < probabilityWordConcept[word].length; conceptInd++) 
						{
							total += 1.0*topicProbabilities[topicInd] *probabilityWordConcept[word][conceptInd]/prowcsum[word]
									* (this.beta + typeTopicCounts[numProbabilityWordConcept[word][conceptInd]][topicInd])
									 /(vBeta + tokensPerTopic[topicInd]);
						}		
					}
				}
				else {
					for(int topicInd= 0; topicInd< numTopics; topicInd++) {
						total += topicProbabilities[topicInd] * (this.beta + typeTopicCounts[numTypes+nonConceptWords[word]][topicInd]) 
								/(vBeta + tokensPerTopic[topicInd]);
					}
				}
				logPerplexity -= (Math.log(total)/logTwo);		
			}
		}
		double output = Math.pow(2, (1.0*logPerplexity/numTokens));
		System.out.println("\t\tPerplexity: "+output);
		return output;
	}
	public void computePhi() {
		for (int i = 0; i < numTopics; i++) {
			Arrays.fill(phi[i], 0.0);
			for (int j = 0; j < numWords; j++) {
				if(nonConceptWords[j]==-1)
				{
					for (int t = 0; t < probabilityWordConcept[j].length; t++) {
						if(typeTopicCounts[numProbabilityWordConcept[j][t]][i]>0)
							phi[i][j]+=1.0*(this.beta + typeTopicCounts[numProbabilityWordConcept[j][t]][i])
									/(vBeta + tokensPerTopic[i])*probabilityWordConcept[j][t]/sumc[numProbabilityWordConcept[j][t]];
					}
				}
				else {
					phi[i][j] = 1.0*(this.beta + typeTopicCounts[numTypes+nonConceptWords[j]][i])/(vBeta + tokensPerTopic[i]);
				}
			}
		}

	}
		
	public int[][] getDocTopicCounts(){
		return docTopicCounts;
	}
	
	public int[][] getTypeTopicCounts(){
		return typeTopicCounts;
	}

	public int[] getTokensPerTopic(){
		return tokensPerTopic;
	}

	public void printTopWords (int nWords) throws IOException
	{
		class WordProb implements Comparable {
			int wi;
			double p;
			public WordProb (int wi, double p) { this.wi = wi; this.p = p; }
			@Override
			public final int compareTo (Object o2) {
				if (p > ((WordProb)o2).p)
					return -1;
				else if (p == ((WordProb)o2).p)
					return 0;
				else return 1;
			}
		}
		
		File topwordsFile = new File("data/topwords_clda"+numTopics); // this is the file for labels
		Writer topwordsWriter = new OutputStreamWriter(new FileOutputStream(topwordsFile));
		
		File phiFile = new File("data/phi_clda"+numTopics); // this is the file for labels
		Writer phiWriter = new OutputStreamWriter(new FileOutputStream(phiFile));
		
		File typeFile = new File("data/type_clda"+numTopics); // this is the file for labels
		Writer typeWriter = new OutputStreamWriter(new FileOutputStream(typeFile));
		
		WordProb[] wp = new WordProb[numWords];
		WordProb[] conceptp = new WordProb[numTypes+numWords];
		for (int ti = 0; ti < numTopics; ti++) {
			int nci = 0;
			//Arrays.fill(sortedTypes, 0);
			for (; nci < numTypes; nci++) {
				conceptp[nci]=new WordProb(nci, typeTopicCounts[nci][ti]);
			}
			for (int aci = 0; aci < numWords; aci++) {
				if(nonConceptWords[aci]!=-1)
					conceptp[nci+aci]=new WordProb(numTypes+aci, typeTopicCounts[numTypes+nonConceptWords[aci]][ti]);
				else {
					conceptp[nci+aci]=new WordProb(numTypes+aci, 0);
				}
			}
			Arrays.sort(conceptp);
			typeWriter.write(ti + "\t");
			for (int i=0; i < nWords; i++) {
				if (conceptp[i].p== 0) { break; }
				if(conceptp[i].wi>=numTypes)
				{
					typeWriter.write(ilist.getDataAlphabet().lookupObject(conceptp[i].wi-numTypes) + " 0\t");
				}
				else {
					typeWriter.write(truevocConcepts[conceptp[i].wi] + " 1\t");
				}
				
			}
			typeWriter.write("\n");
			for (int wi = 0; wi < numWords; wi++)
			{
				double p=0;
				if(nonConceptWords[wi]==-1)
				{
					for (int ci = 0; ci < probabilityWordConcept[wi].length; ci++) {
						p += 1.0*probabilityWordConcept[wi][ci]/sumc[numProbabilityWordConcept[wi][ci]] * typeTopicCounts[numProbabilityWordConcept[wi][ci]][ti] / tokensPerTopic[ti];
					}
				}
				else {
					p=((double)typeTopicCounts[numTypes+nonConceptWords[wi]][ti]) / tokensPerTopic[ti];
				}
				phiWriter.write(Double.toString(p)+"\t");
				wp[wi] = new WordProb (wi, p);
			}
			phiWriter.write("\n");
			Arrays.sort (wp);
			topwordsWriter.write(ti+"\t");
			for (int i = 0; i < nWords; i++)
				topwordsWriter.write(ilist.getDataAlphabet().lookupObject(wp[i].wi).toString() + " ");
			topwordsWriter.write("\n");
		}
		phiWriter.close();
		topwordsWriter.close();
		typeWriter.close();
	}

	public void printDocumentTopics (File f) throws IOException
	{
		printDocumentTopics (new PrintWriter (new FileWriter (f)));
	}

	public void printDocumentTopics (PrintWriter pw) {
		printDocumentTopics (pw, 0.0, -1);
	}

	public void printDocumentTopics (PrintWriter pw, double threshold, int max)
	{
		pw.println ("#doc source topic proportion ...");
		int docLen;
		double topicDist[] = new double[topics.length];
		for (int di = 0; di < topics.length; di++) {
			pw.print (di); pw.print (' ');
			if (ilist.get(di).getSource() != null){
				pw.print (ilist.get(di).getSource().toString()); 
			}
			else {
				pw.print("null-source");
			}
			pw.print (' ');
			docLen = topics[di].length;
			for (int ti = 0; ti < numTopics; ti++)
				topicDist[ti] = (((float)docTopicCounts[di][ti])/docLen);
			if (max < 0) max = numTopics;
			for (int tp = 0; tp < max; tp++) {
				double maxvalue = 0;
				int maxindex = -1;
				for (int ti = 0; ti < numTopics; ti++)
					if (topicDist[ti] > maxvalue) {
						maxvalue = topicDist[ti];
						maxindex = ti;
					}
				if (maxindex == -1 || topicDist[maxindex] < threshold)
					break;
				pw.print (maxindex+" "+topicDist[maxindex]+" ");
				topicDist[maxindex] = 0;
			}
			pw.println (' ');
		}
	}



	public void printState (File f) throws IOException
	{
		PrintWriter writer = new PrintWriter (new FileWriter(f));
		printState (writer);
		writer.close();
	}

	public void assignment () throws IOException
	{
		File topicassignmentFile = new File("data/topicassignment"+numTopics); // this is the file for labels
		Writer topicassignmentWriter = new OutputStreamWriter(new FileOutputStream(topicassignmentFile));
		File typeassignmentFile = new File("data/typeassignment"+numTopics); // this is the file for labels
		Writer typeassignmentWriter = new OutputStreamWriter(new FileOutputStream(typeassignmentFile));
		File wordsFile = new File("data/words"+numTopics); // this is the file for labels
		Writer wordsWriter = new OutputStreamWriter(new FileOutputStream(wordsFile));
		
		for (int i = 0; i < ilist.size(); i++) {
			FeatureSequence fs = (FeatureSequence) ilist.get(i).getData();
			for (int j = 0; j < fs.getLength(); j++) {
				topicassignmentWriter.write(topics[i][j]+" ");
				int wordId=fs.getIndexAtPosition(j);
				if(nonConceptWords[wordId]==-1)
				{
					typeassignmentWriter.write("c"+concepts[i][j]+" ");
				}
				else {
					typeassignmentWriter.write("w"+fs.getIndexAtPosition(j)+" ");
				}
				wordsWriter.write(fs.getIndexAtPosition(j)+" ");
			}
			topicassignmentWriter.write("\n");
			typeassignmentWriter.write("\n");
			wordsWriter.write("\n");
		}
		
		typeassignmentWriter.close();
		topicassignmentWriter.close();
		wordsWriter.close();
	}

public void printState (PrintWriter pw)
{
	  Alphabet a = ilist.getDataAlphabet();
	  pw.println ("#doc pos typeindex type topic");
	  for (int di = 0; di < topics.length; di++) {
		  FeatureSequence fs = (FeatureSequence) ilist.get(di).getData();
		  for (int si = 0; si < topics[di].length; si++) {
			  int type = fs.getIndexAtPosition(si);
			  pw.print(di); pw.print(' ');
			  pw.print(si); pw.print(' ');
			  pw.print(type); pw.print(' ');
			  pw.print(a.lookupObject(type)); pw.print(' ');
			  pw.print(topics[di][si]); pw.println();
		  }
	  }
}

public void write (File f) {
 try {
   ObjectOutputStream oos = new ObjectOutputStream (new FileOutputStream(f));
   oos.writeObject(this);
   oos.close();
 }
 catch (IOException e) {
   System.err.println("Exception writing file " + f + ": " + e);
 }
}


// Serialization

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;

	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject (ilist);
		out.writeInt (numTopics);
		out.writeDouble (alpha);
		out.writeDouble (beta);
		out.writeDouble (tAlpha);
		out.writeDouble (vBeta);
		for (int di = 0; di < topics.length; di ++)
			for (int si = 0; si < topics[di].length; si++)
				out.writeInt (topics[di][si]);
		for (int di = 0; di < topics.length; di ++)
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt (docTopicCounts[di][ti]);
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt (typeTopicCounts[fi][ti]);
		for (int ti = 0; ti < numTopics; ti++)
			out.writeInt (tokensPerTopic[ti]);
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int featuresLength;
		int version = in.readInt ();
		ilist = (InstanceList) in.readObject ();
		numTopics = in.readInt();
		alpha = in.readDouble();
		beta = in.readDouble();
		tAlpha = in.readDouble();
		vBeta = in.readDouble();
		int numDocs = ilist.size();
		topics = new int[numDocs][];
		for (int di = 0; di < ilist.size(); di++) {
			int docLen = ((FeatureSequence)ilist.get(di).getData()).getLength();
			topics[di] = new int[docLen];
			for (int si = 0; si < docLen; si++)
				topics[di][si] = in.readInt();
		}
		docTopicCounts = new int[numDocs][numTopics];
		for (int di = 0; di < ilist.size(); di++)
			for (int ti = 0; ti < numTopics; ti++)
				docTopicCounts[di][ti] = in.readInt();
		int numTypes = ilist.getDataAlphabet().size();
		typeTopicCounts = new int[numTypes][numTopics];
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				typeTopicCounts[fi][ti] = in.readInt();
		tokensPerTopic = new int[numTopics];
		for (int ti = 0; ti < numTopics; ti++)
			tokensPerTopic[ti] = in.readInt();
	}

	public InstanceList getInstanceList ()
	{
		return ilist;
	}

	// Recommended to use mallet/bin/vectors2topics instead.
	public static void main (String[] args) throws IOException
	{
		InstanceList ilist = InstanceList.load (new File(args[0]));
		int numIterations = 1000;
		int topic = args.length > 1 ? Integer.parseInt(args[1]) : 10;
		int numTopWords = args.length > 2 ? Integer.parseInt(args[2]) : 10;
		System.out.println ("Conceptualization LDA Data loaded.");
		CLDA conceptualizationLDA = new CLDA (topic);
		conceptualizationLDA.estimate (ilist, numIterations, 50, 0, null, new Randoms());  // should be 1100
		conceptualizationLDA.printTopWords (numTopWords);
		conceptualizationLDA.assignment ();
		conceptualizationLDA.printDocumentTopics (new File(args[0]+topic+".clda"));
	}

}
