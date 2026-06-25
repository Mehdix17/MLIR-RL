const fs = require('fs');
const path = require('path');

const projectRoot = '/scratch/mb10856/MLIR-RL';
const uaDir = path.join(projectRoot, '.understand-anything');
const timestamp = '20260624_043106';
const trashDir = path.join(uaDir, `.trash-${timestamp}`);
const trashIntermediate = path.join(trashDir, 'intermediate');
const trashTmp = path.join(trashDir, 'tmp');

try {
  // Create trash directories
  fs.mkdirSync(trashIntermediate, { recursive: true });
  fs.mkdirSync(trashTmp, { recursive: true });

  // Move intermediate files (except scan-result.json)
  const intermediateDir = path.join(uaDir, 'intermediate');
  const intermediateFiles = fs.readdirSync(intermediateDir);
  intermediateFiles.forEach(file => {
    if (file === 'scan-result.json') return;
    const src = path.join(intermediateDir, file);
    const dest = path.join(trashIntermediate, file);
    fs.renameSync(src, dest);
  });

  // Move tmp files (except cleanup.js)
  const tmpDir = path.join(uaDir, 'tmp');
  const tmpFiles = fs.readdirSync(tmpDir);
  tmpFiles.forEach(file => {
    if (file === 'cleanup.js') return;
    const src = path.join(tmpDir, file);
    const dest = path.join(trashTmp, file);
    fs.renameSync(src, dest);
  });

  console.log('Cleanup completed successfully.');
} catch (err) {
  console.error('Error during cleanup:', err.message);
  process.exit(1);
}
