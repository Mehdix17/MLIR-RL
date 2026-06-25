const fs = require('fs');
const path = require('path');

const projectRoot = '/scratch/mb10856/MLIR-RL';
const scanResultPath = path.join(projectRoot, '.understand-anything/intermediate/scan-result.json');
const outputPath = path.join(projectRoot, '.understand-anything/intermediate/fingerprint-input.json');

try {
  const scanResult = JSON.parse(fs.readFileSync(scanResultPath, 'utf8'));
  const filePaths = scanResult.files.map(f => f.path);
  const gitCommitHash = '84a14bbe2558eef15717bca5fa539ce812f1fc5f';
  
  const fingerprintInput = {
    projectRoot: projectRoot,
    sourceFilePaths: filePaths,
    gitCommitHash: gitCommitHash
  };
  
  fs.writeFileSync(outputPath, JSON.stringify(fingerprintInput, null, 2));
  console.log(`Successfully wrote fingerprint-input.json with ${filePaths.length} files.`);
} catch (err) {
  console.error('Error preparing fingerprint input:', err.message);
  process.exit(1);
}
