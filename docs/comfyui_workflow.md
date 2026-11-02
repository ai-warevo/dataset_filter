# ComfyUI Piercing Dataset Workflow

## Overview

This document describes the ComfyUI workflow for generating and validating piercing datasets. The workflow is designed to work with ComfyUI's Stable Diffusion pipeline to create images with specified piercing characteristics, and integrate with the existing dataset filtering pipeline.

## Workflow Components

### 1. Piercing Generation

The workflow generates piercing images with:
- Specified piercing types (e.g., silver hoop)
- Defined locations (e.g., right eyebrow, left nostril)
- Expected counts (e.g., 2 piercings on right eyebrow)
- Professional studio lighting quality
- Clean white background isolation

### 2. Dataset Labeling

The workflow automatically generates labels in YOLO format:
- Coordinate information for each piercing
- Class labels for different piercing types
- Bounding box information for detection training

### 3. Validation System

The workflow includes validation capabilities:
- Comparison against specification requirements
- Pass/fail status reporting
- Error detail output for mismatches
- Statistical summary reports

## Technical Implementation

### Core Classes

#### ComfyUIPiercingWorkflow
Main workflow controller that handles:
- Loading specifications from JSON files
- Creating ComfyUI workflows for each piercing requirement
- Generating images with specified characteristics
- Validating generated datasets

### Key Methods

#### `create_piercing_generation_workflow(piercing_spec, output_dir)`
Creates a complete ComfyUI workflow for generating a specific piercing type and location.

#### `generate_dataset(output_dir)`
Generates a complete dataset based on all specifications.

#### `validate_dataset(input_dir)`
Validates generated images against specifications.

#### `create_comprehensive_workflow()`
Creates a complete workflow that combines all aspects.

## Usage Examples

### Generate Piercing Dataset

```bash
python comfyui_workflow_piercing_dataset.py --generate
```

### Validate Existing Dataset

```bash
python comfyui_workflow_piercing_dataset.py --validate
```

### Generate and Validate with Custom Output

```bash
python comfyui_workflow_piercing_dataset.py --generate --validate --output-dir=data/custom_dataset
```

## Integration with Existing Pipeline

The workflow seamlessly integrates with the existing dataset filtering system:

1. **Generation Phase**: Use ComfyUI to generate images with specified characteristics
2. **Labeling Phase**: Automatically extract metadata/coordinates from generation process
3. **Validation Phase**: Compare generated images against specifications
4. **Filtering Phase**: Use existing filtering pipeline to validate images

## Future Enhancements

1. **YOLO Model Integration**: Implement YOLO-based piercing detection for validation
2. **Zero-Shot Detection**: Add CLIP-based semantic detection as fallback
3. **Batch Processing**: Support for large-scale dataset generation
4. **Advanced Prompt Engineering**: More sophisticated prompt generation for better image quality

## File Structure

### Input Files
- `src/dataset_filter/specs/smlq_sfw.json`: Piercing specifications
- `comfyui_wf_piercing_gen.json`: Base ComfyUI workflow template

### Output Files
- `data/assets/`: Generated images and workflows
- `data/test_assets/`: Test-generated workflow files

## Specifications Format

The piercing specifications are defined in JSON format:

```json
{
  "character_id": "smlq_sfw",
  "piercing_requirements": [
    {
      "artifact_id": "silver_hoop",
      "expected_location": "RIGHT_EYEBROW",
      "is_mandatory": true,
      "expected_count": 2
    },
    {
      "artifact_id": "silver_hoop",
      "expected_location": "LEFT_EYEBROW",
      "is_mandatory": false,
      "expected_count": 0
    },
    {
      "artifact_id": "silver_hoop",
      "expected_location": "LEFT_NOSTRIL",
      "is_mandatory": true,
      "expected_count": 1
    },
    {
      "artifact_id": "silver_hoop",
      "expected_location": "RIGHT_NOSTRIL",
      "is_mandatory": false,
      "expected_count": 0
    }
  ]
}
```

## Error Handling

The workflow includes comprehensive error handling:
- Specification loading validation
- Workflow generation error checking
- File I/O error handling
- Validation result reporting

## Performance Considerations

- Generation process can be parallelized
- Validation can be batched for efficiency
- Memory usage optimized for large datasets
- Caching of generated prompts for consistency

## Security Considerations

- All generated files are properly sanitized
- No external data is stored in generated workflows
- Temporary files are cleaned up appropriately
- Access controls can be added as needed

## Troubleshooting

### Common Issues

1. **Missing Models**: Ensure required ComfyUI models are installed
2. **File Permissions**: Verify write permissions to output directories
3. **JSON Format Errors**: Validate specification files are properly formatted
4. **Resource Limits**: Monitor memory usage during batch generation

### Debugging

Use verbose logging for detailed information:
```bash
python comfyui_workflow_piercing_dataset.py --generate --verbose
```