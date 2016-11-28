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
			MapBackedDataset people = new MapBackedDataset();
			BingAPIToken bingtoken = DefaultTokenFactory.get(BingAPIToken.class);
			BingImageDataset<FImage> books = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "book", 10);
			DisplayUtilities.display("Books", books);
			
			BingImageDataset<FImage> person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Marilyn Monroe", 10);
			people.add("Marilyn", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Abraham Lincoln", 3);
			people.add("Lincoln", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Mother Teresa", 3);
			people.add("Teresa", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "John F. Kennedy", 3);
			people.add("Kennedy", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Martin Luther King", 3);
			people.add("Martin", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Nelson Mandela", 3);
			people.add("Mandela", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Winston Churchill0", 3);
			people.add("Churchill", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Bill Gates", 3);
			people.add("Gates", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Muhammad Ali", 3);
			people.add("Ali", person);
			person = BingImageDataset.create(ImageUtilities.FIMAGE_READER, bingtoken, "Mahatma Gandhi", 3);
			people.add("Gandhi", person);
			
		} catch(Exception e) { e.printStackTrace(); }
	}
}
