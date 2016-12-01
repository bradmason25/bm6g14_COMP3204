package uk.ac.soton.ecs.bm6g14.ch14;

import java.util.ArrayList;
import java.util.List;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.time.Timer;
import org.openimaj.util.function.Operation;
import org.openimaj.util.parallel.Parallel;


public class Tutorial14_1_1 {
	public static void main(String[] args) {
		try {
			
			VFSGroupDataset<MBFImage> allImages = Caltech101.getImages(ImageUtilities.MBFIMAGE_READER);
			
			GroupedDataset<String, ListDataset<MBFImage>, MBFImage> images = GroupSampler.sample(allImages, 8, false);
			
			List<MBFImage> output = new ArrayList<MBFImage>();
			ResizeProcessor resize = new ResizeProcessor(200);
			
			Timer t1 = Timer.timer();
			for (ListDataset<MBFImage> clzImages : images.values()) {
			    MBFImage current = new MBFImage(200, 200, ColourSpace.RGB);

			    for (MBFImage i : clzImages) {
			        MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);
			        tmp.fill(RGBColour.WHITE);

			        MBFImage small = i.process(resize).normalise();
			        int x = (200 - small.getWidth()) / 2;
			        int y = (200 - small.getHeight()) / 2;
			        tmp.drawImage(small, x, y);

			        current.addInplace(tmp);
			    }
			    current.divideInplace((float) clzImages.size());
			    output.add(current);
			}
			System.out.println("No Parallel Time: " + t1.duration() + "ms");
			
			Timer t2 = Timer.timer();
			
			final List<MBFImage> output2 = new ArrayList<MBFImage>();
			final ResizeProcessor resize2 = new ResizeProcessor(200);
			
			Parallel.forEach(images.values(), new Operation<ListDataset<MBFImage>>() {
				public void perform(ListDataset<MBFImage> clzImages) {
					MBFImage current = new MBFImage(200,200,ColourSpace.RGB);
					
					for(MBFImage i: clzImages) {
				        MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);
				        tmp.fill(RGBColour.WHITE);

				        MBFImage small = i.process(resize2).normalise();
				        int x = (200 - small.getWidth()) / 2;
				        int y = (200 - small.getHeight()) / 2;
				        tmp.drawImage(small, x, y);

				        current.addInplace(tmp);
					}
					current.divideInplace((float) clzImages.size());
					
					synchronized(output2) {
						output2.add(current);
					}
				}
			});
			System.out.println("Outer Parallel Time: " + t2.duration() + "ms");

			Timer t3 = Timer.timer();
			final List<MBFImage> output3 = new ArrayList<MBFImage>();
			final ResizeProcessor resize3 = new ResizeProcessor(200);
			
			for (ListDataset<MBFImage> clzImages : images.values()) {
			    final MBFImage current = new MBFImage(200, 200, ColourSpace.RGB);

			    Parallel.forEach(clzImages, new Operation<MBFImage>() {
			        public void perform(MBFImage i) {
			            final MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);
			            tmp.fill(RGBColour.WHITE);

			            final MBFImage small = i.process(resize3).normalise();
			            final int x = (200 - small.getWidth()) / 2;
			            final int y = (200 - small.getHeight()) / 2;
			            tmp.drawImage(small, x, y);

			            synchronized (current) {
			                current.addInplace(tmp);
			            }
			        }
			    });
			    current.divideInplace((float) clzImages.size());
			    output3.add(current);
			}
			System.out.println("Inner Parallel Time: " + t3.duration() + "ms");
			
			
			//By parallelising the outer for loop you make it faster than the original code but not faster than the
			//parallel inner for loop. This is probably because most of the processing is done within the inner loop
			//which wouldn't alter the performance between them but then the synchronised block restricts access to
			//a major object. This cost of this is greater than the synchronised access to the temporary object current
			//within the inner loop.
			
			
			
			DisplayUtilities.display("Images", output);
		} catch(Exception e) {}
	}
}
