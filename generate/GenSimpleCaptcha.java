import java.io.*;
import java.awt.*;
import java.awt.Color;
import java.awt.image.*;
import javax.imageio.*;
import nl.captcha.*;
import nl.captcha.backgrounds.*;
import nl.captcha.gimpy.*;
import nl.captcha.noise.*;

import java.util.Random;

public class GenSimpleCaptcha {

    public static int min = 0;
    public static int max = 255;
    public static Random rand = new Random();

    public static Color rColor(){
        int r = rand.nextInt((max - min) + 1) + min;
        int g = rand.nextInt((max - min) + 1) + min;
        int b = rand.nextInt((max - min) + 1) + min;
        return new Color(r, g, b);
    }

    public static Captcha new_captcha(){
        return new Captcha.Builder(200, 50)
        .addText()
        .addBackground(new GradiatedBackgroundProducer(rColor(), rColor()))
        .gimp(new FishEyeGimpyRenderer())
        .addNoise(new StraightLineNoiseProducer(Color.black, 4))
        .addBorder()
        .build(); 
    }

    public static void build_new_image(int i){
        Captcha captcha = new_captcha();
        String ans = captcha.getAnswer();
        BufferedImage img = captcha.getImage();
        File f = new File("./images/" + i + "_" + ans + ".png");

        try{
            ImageIO.write(captcha.getImage(), "PNG", f);
        } catch(Exception e){
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // Generate and save images.
        for (int i = 0; i < 30000; i++)
            build_new_image(i);

        System.out.println("Done!!");
    }
}
