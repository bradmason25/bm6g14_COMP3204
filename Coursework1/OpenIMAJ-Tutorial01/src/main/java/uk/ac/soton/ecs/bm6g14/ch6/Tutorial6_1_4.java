package uk.ac.soton.ecs.bm6g14.ch6;

import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.dataset.BingImageDataset;
import org.openimaj.util.api.auth.DefaultTokenFactory;
import org.openimaj.util.api.auth.common.BingAPIToken;

public class Tutorial6_1_4 {
	public static void main(String[] args) {
		try {
			BingAPIToken bingtoken = DefaultTokenFactory.get(BingAPIToken.class);
			BingImageDataset<FImage> books = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "book", 10);
			DisplayUtilities.display("Books", books);
			
			
			BingImageDataset<FImage> person1 = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Marilyn Monroe", 3);
			BingImageDataset<FImage> person2 = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Mother Teresa", 3);
			BingImageDataset<FImage> person3 = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "John F. Kennedy", 3);
			BingImageDataset<FImage> person4 = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Martin Luther King", 3);
			BingImageDataset<FImage> person5 = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Nelson Mandela", 3);
			
			MapBackedDataset<String, BingImageDataset<FImage>, FImage> people = MapBackedDataset.of(person1, person2, person3, person4, person5);
			
			
		} catch(Exception e) { e.printStackTrace(); }
	}
}
