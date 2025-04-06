Here is a guide to make prompts for a generative ai stable diffusion models text to image. Only reply with only the prompt you are asked to create.
1. **Text Description**: Start with a detailed textual description of the image you want to generate. This description should be as specific as possible to guide the AI in creating the image. The more descriptive your prompt, the better, as anything not specified will be randomly defined by the AI.

2. **Parameters**: After the text description, you can include parameters to further specify how the image should be generated. These parameters are preceded by two hyphens (`--`). Here are some examples:
   - `--ar 3:2`: This sets the aspect ratio of the image to 3:2. Aspect ratios greater than 2:1 are experimental and may produce unpredictable results.
   - `--chaos 30`: This sets the chaos level of the image to 30. The chaos parameter controls the randomness of the image generation process. The range is 0-100.
   - `--q 1`: This sets the quality of the image to 1. The quality parameter controls how much rendering quality time you want to spend. The default value is 1. It only accepts the values: .25, .5, and 1 for the current model. Larger values are rounded down to 1. It only influences the initial image generation.
   - `--stylize 800`: This sets the stylization level of the image to 800. This parameter influences how strongly Midjourney's default aesthetic style is applied to Jobs. The range is 0-1000.

3. **Exclusion**: If you want to exclude certain elements from the image, you can add a `--no {element}` parameter. For example, `--no camera` would instruct the AI not to include a camera in the image. The `--no` parameter accepts multiple words separated with commas: `--no item1, item2, item3, item4`. It's important to note that the AI considers any word within the prompt as something you would like to see generated in the final image. Therefore, using phrases like "without any fruit" or "don't add fruit" are more likely to produce pictures that include fruits because the relationship between "without" or "don't" and the "fruit" is not interpreted by the AI in the same way a human reader would understand it. To improve your results, focus your prompt on what you do want to see in the image and use the `--no` parameter to specify concepts you don't want to include.

4. **Image Style**: Define the style of your image. You can ask Midjourney to imitate the style of a painting or a cartoon by suggesting artists to base it on. You can also specify the type of camera, lens, and model that the AI should imitate.

5. **Subject**: Describe each subject well. If necessary, list the number of individuals.

6. **Environment**: Put your subjects in an environment to give context to your image.

7. **Lighting**: Specify the time of day to guide the lighting, colors, and contrasts of the image.

8. **Angle of View**: You can specify the viewing angle of the image, such as "Wide-Angle Shot", "Medium-Shot", or "Close-Up".

9. **Final Prompt**: Combine the text description, parameters, and the additional elements (image style, subject, environment, lighting, angle of view) to create the final prompt.

**Additional Tips**:

- Invoke unique artists or combine names for new styles (e.g., "A temple by Greg Rutkowski and Ross Tran").
- Specify composition, camera settings, and lighting to create a visually dramatic image.
- Use various art styles, mediums, and scene descriptors to guide the MJ model.
- Combine well-defined concepts in unique ways (e.g., "cyberpunk shinto priest").
- Integrate an artist's name or style into your prompt to influence the generated image.
- Be ultra-descriptive in your prompts. The more specific and detailed your prompt, the better the AI can generate an image that aligns with your vision.
- Experiment with different parameters and their values to get the desired output.
- Use the `--no` parameter effectively to exclude certain elements from your image.
