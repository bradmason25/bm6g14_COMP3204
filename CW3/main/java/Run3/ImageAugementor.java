package Run3;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map.Entry;

import javax.imageio.ImageIO;
import javax.swing.JFileChooser;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.processing.transform.AffineSimulation;

public class ImageAugementor {
	String targetDir = "";
	/*
	 * Asks the user to provide a root directory and then augments all the images within sub-directories
	 * Creating: zoomed/rotated/cropped/flipped images
	 */
	public static void main(String args[])
	{
		new ImageAugementor();
	}
	ImageAugementor()
	{
		getImages();
		augmentImages();
	}
	void getImages()
	{
		final JFileChooser fc = new JFileChooser();
		fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		int returnVal = fc.showOpenDialog(null);
		if(returnVal == JFileChooser.APPROVE_OPTION)
			targetDir = fc.getSelectedFile().getAbsolutePath();
		System.out.println(returnVal);
	}
	void augmentImages()
	{
		VFSGroupDataset<FImage> imageSet = null;
		try {
			imageSet = new VFSGroupDataset<FImage>( targetDir, ImageUtilities.FIMAGE_READER);
		} catch (FileSystemException e) {
			e.printStackTrace();
		}
		if(imageSet==null)
			return;
		int index = 0;
		int augtype = 0;
		for (Entry<String, VFSListDataset<FImage>> entry : imageSet.entrySet()) {
			index = 0;
			for(FImage image : entry.getValue())
			{
				augtype = 0;
				for(FImage augimg : getAugList(image))
				{
					File newImgFile = new File(targetDir+"/"+entry.getKey()+"/aug_"+index+"_"+augtype+".jpg");
					System.out.println("Trying to save: aug_"+index+"_"+augtype+" to:: "+newImgFile.getPath());
					try {
						ImageIO.write(ImageUtilities.createBufferedImage(augimg), "jpg", newImgFile);
					} catch (IOException e) {
						System.out.println("Failed to save img: aug_"+index+"_"+augtype);
					}
					System.out.println("fileExists:"+newImgFile.exists());
					augtype++;
				}
				index++;
			}
		}
	}
	
	ArrayList<FImage> getAugList(FImage img)
	{
		ArrayList<FImage> imgs = new ArrayList<FImage>();
		imgs.add(randomCrop(img));
		imgs.add(randomFlip(img));
		imgs.add(slightRotation(img));
		imgs.add(randomZoom(img));
		return imgs;
	}
	
	FImage randomCrop(FImage img)
	{
		img = img.clone();
		int cropX = (int)(Math.random()*(img.getWidth()*0.2));
		int cropY = (int)(Math.random()*(img.getHeight()*0.2));	
		
		return img.extractROI(cropX, cropY, img.getWidth()-cropX*2, img.getHeight()-cropY*2);
	}
	FImage slightRotation(FImage img)
	{
		img = img.clone();
		float angle = (float) (-0.2f+Math.random()*0.4f);
		return AffineSimulation.transformImage(img, angle, 1);
	}
	FImage randomFlip(FImage img)
	{
		img = img.clone();
		img.flipX();
		return img;
	}
	FImage randomZoom(FImage img)
	{
		img = img.clone();
		float amount = (float) (0.75f+Math.random()*0.5f);
		ResizeProcessor proc = new ResizeProcessor(amount);
		proc.processImage(img);
		return img;
	}
	
}
