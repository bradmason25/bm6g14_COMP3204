package Run3;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.time.Timer;
import org.openimaj.util.pair.IntFloatPair;

import com.google.common.base.Strings;

import de.bwaldvogel.liblinear.SolverType;

/*
 * Classifies images by taking densely sampled pixel patches, then groups them into a bag of visual words, and producing a histogram for each image
 */
public class DenselySampledPixelPatches {
	
	// Uses the given assigner to generate a histogram for the given image
	class DenselySampledPixelPatchExtractor implements FeatureExtractor<SparseIntFV, FImage> {
		HardAssigner<float[], float[], IntFloatPair> assigner;
		int binSize, stepSize;
		
		public DenselySampledPixelPatchExtractor( HardAssigner<float[], float[], IntFloatPair> assigner, int binSize,
				int stepSize ) {
			this.assigner = assigner;
			this.binSize = binSize;
			this.stepSize = stepSize;
		}
		
		public SparseIntFV extractFeature(FImage object ) {
			FImage image = object;
			
			List<float[]> imageFeatureVectors = getDenselySampledPixelPatches( image, binSize, stepSize );
			
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>( assigner );
			
			return bovw.aggregateVectorsRaw( imageFeatureVectors );
		}
	}
	
	static final File cacheLocation = new File( "DenselySampledPixelPatchCache" );
	
	PrintWriter log;
	Timer timer;
	
	String initialiseLog() throws FileNotFoundException, UnsupportedEncodingException {
		timer = new Timer();
		timer.start();
		String now = String.valueOf( new Date().getTime() );
		log = new PrintWriter( "logs/run2log_" + now + ".txt", "UTF-8" );
		
		logln( "starting at " + new Date().toString() );
		
		return now;
	}
	
	void closeLog() {
		logln( "terminating" );
		log.close();
	}
	
	void logln( String txt ) {
		try {
			if ( log == null ) {
				initialiseLog();
			}
			
			txt = Strings.padStart( String.valueOf( timer.duration() ), 10, ' ' ) + " " + txt;
			log.println( txt );
			log.flush();
			System.out.println( txt );
		} catch ( FileNotFoundException e ) {
			e.printStackTrace();
		} catch ( UnsupportedEncodingException e ) {
			e.printStackTrace();
		}
	}
	
	// Return all vectors
	List<float[]> getDenselySampledPixelPatches( FImage img, int binSize, int stepSize ) {
		return getDenselySampledPixelPatches( img, binSize, stepSize, -1 );
	}
	
	// Produces a list of vectors based on densely sampled pixel patches from
	// the given image.
	// vectorsToReturn may be negative to return all vectors
	List<float[]> getDenselySampledPixelPatches( FImage img, int binSize, int stepSize, int vectorsToReturn ) {
		
		List<float[]> denselySampledPixelPatches = new ArrayList<>();
		
		int stepCountX = ( img.width - binSize ) / stepSize + 1;
		int stepCountY = ( img.height - binSize ) / stepSize + 1;
		
		// For each patch:
		for ( int yStep = 0; yStep < stepCountY; ++yStep ) {
			int topEdge = yStep * stepSize;
			for ( int xStep = 0; xStep < stepCountX; ++xStep ) {
				int leftEdge = xStep * stepSize;
				
				float[] fv = calculatePatch( img, binSize, topEdge, leftEdge );
				
				denselySampledPixelPatches.add( fv );
			}
		}
		
		// Take a sub-list if necessary
		if ( vectorsToReturn >= 0 && denselySampledPixelPatches.size() > vectorsToReturn ) {
			Collections.shuffle( denselySampledPixelPatches );
			denselySampledPixelPatches = denselySampledPixelPatches.subList( 0, vectorsToReturn );
		}
		
		return denselySampledPixelPatches;
	}
	
	// Calculate the vector from a specific patch on an image
	float[] calculatePatch( FImage img, int binSize, int topEdge, int leftEdge ) {
		float[] fv = new float[ binSize * binSize ];
		
		// Populate the vector based on the pixels inside the patch
		
		// For each pixel within the patch:
		for ( int yOffset = 0; yOffset < binSize; ++yOffset ) {
			int y = topEdge + yOffset;
			for ( int xOffset = 0; xOffset < binSize; ++xOffset ) {
				int x = leftEdge + xOffset;
				
				fv[ yOffset * binSize + xOffset ] = img.getPixel( x, y );
			}
		}
		
		// normalise the densely sampled pixel patch
		float sum = 0;
		float xSquaredSum = 0;
		for ( float pixelValue : fv ) {
			sum += pixelValue;
			xSquaredSum += pixelValue * pixelValue;
		}
		float mean = sum / fv.length;
		double stdev = Math.sqrt( xSquaredSum / fv.length - mean * mean );
		for ( int i = 0; i < fv.length; ++i ) {
			fv[ i ] = (float) ( ( fv[ i ] - mean ) / stdev );
		}
		
		return fv;
	}
	
	
	// Calculates all densely sampled pixel patches in the dataset, and returns
	// a sample of the specified size
	List<float[]> samplePixelPatches(
			GroupedDataset<String, ListDataset<FImage>, FImage> dataset,
			int binSize, int stepSize, int sampleSize ) {
		
		List<float[]> allDenselySampledPixelPatches = new ArrayList<>();
		
		// Sample the vectors after processing each image due to memory
		// issues
		int vectorsPerTrainingImage = sampleSize / dataset.numInstances() + 1;
		logln( "returning " + vectorsPerTrainingImage + " vectors per training image" );
		
		for ( FImage idImg : dataset ) {
			FImage img = idImg;
			
			allDenselySampledPixelPatches
					.addAll( getDenselySampledPixelPatches( img, binSize, stepSize, vectorsPerTrainingImage ) );
		}
		
		logln( "Calculated the densely sampled pixel patches (" + allDenselySampledPixelPatches.size() + ")" );
		
		// Limit the number of vectors to a specified amount
		List<float[]> fvSample;
		if ( allDenselySampledPixelPatches.size() > sampleSize ) {
			logln( "Using a sample of feature vectors" );
			
			Collections.shuffle( allDenselySampledPixelPatches );
			
			logln( "Shuffled list" );
			
			fvSample = allDenselySampledPixelPatches.subList( 0, sampleSize );
			
			logln( "Taken sublist" );
		} else {
			fvSample = allDenselySampledPixelPatches;
		}
		
		return fvSample;
	}
	
	// Use the vocabulary of visual words and the training data to train an
	// annotator
	LiblinearAnnotator<FImage, String> trainAnnotator( FloatCentroidsResult vocabulary, int binSize,
			int stepSize,
			GroupedDataset<String, ListDataset<FImage>, FImage> trainingData ) {
		
		HardAssigner<float[], float[], IntFloatPair> assigner = vocabulary.defaultHardAssigner();
		
		HomogeneousKernelMap kernelMap = new HomogeneousKernelMap( HomogeneousKernelMap.KernelType.Chi2,
				HomogeneousKernelMap.WindowType.Rectangular );
		FeatureExtractor<DoubleFV, FImage> nonCachedExtractor = kernelMap
				.createWrappedExtractor( new DenselySampledPixelPatchExtractor( assigner, binSize, stepSize ) );
		
		LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<FImage, String>(
				nonCachedExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001 );
		annotator.train( trainingData );
		
		return annotator;
	}

	// Create and train the evaluator based on the given data and parameters
	LiblinearAnnotator<FImage, String> createAnnotator(
			GroupedDataset<String, ListDataset<FImage>, FImage> trainingData,
			int binSize, int stepSize, int vectorSampleSize ) {
		
		List<float[]> fvSample = samplePixelPatches( trainingData, binSize, stepSize, vectorSampleSize );
		
		// Cluster the sample of vectors to form a set of visual words
		FloatKMeans kmeans = FloatKMeans.createKDTreeEnsemble( 500 );
		FloatCentroidsResult vocabulary = kmeans.cluster( fvSample.toArray( new float[ 0 ][] ) );
		
		logln( "Clustered vectors" );
		
		return trainAnnotator( vocabulary, binSize, stepSize, trainingData );
		
	}
	
}
