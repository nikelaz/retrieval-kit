const assert = require('node:assert/strict')

const { RKit } = require('../index.js')

async function main() {
  const rkit = new RKit({
    lanceDbPath: '/tmp/retrieval-kit-node-smoke',
    vectorDimensions: 384,
  })

  const tools = await rkit.getToolDefinitions()
  assert.equal(tools.length, 4)
  assert.deepEqual(
    tools.map((tool) => tool.name),
    ['semantic_search', 'keyword_search', 'list_documents', 'get_document'],
  )
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
