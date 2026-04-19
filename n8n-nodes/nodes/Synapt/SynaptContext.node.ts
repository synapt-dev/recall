import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';

// TODO: synapt server currently exposes stdio MCP transport only.
// These nodes are scaffolded against a planned HTTP transport endpoint.
// See: https://github.com/synapt-dev/recall/issues for tracking.

export class SynaptContext implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Synapt Context',
		name: 'synaptContext',
		icon: 'file:synapt.svg',
		group: ['transform'],
		version: 1,
		description: 'Retrieve file history, timeline, or drill into a search result from recall memory',
		defaults: {
			name: 'Synapt Context',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'synaptApi',
				required: true,
			},
		],
		properties: [
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				default: 'files',
				noDataExpression: true,
				options: [
					{
						name: 'File History',
						value: 'files',
						description: 'Find past sessions that touched a specific file',
					},
					{
						name: 'Timeline',
						value: 'timeline',
						description: 'View chronological timeline of work arcs',
					},
					{
						name: 'Drill Down',
						value: 'context',
						description: 'Get full transcript for a chunk or cluster from a search result',
					},
				],
			},
			{
				displayName: 'File Pattern',
				name: 'pattern',
				type: 'string',
				default: '',
				required: true,
				placeholder: 'src/auth.py',
				description: 'File path or partial path to search for (supports partial matching)',
				displayOptions: {
					show: {
						operation: ['files'],
					},
				},
			},
			{
				displayName: 'Max Chunks',
				name: 'maxChunks',
				type: 'number',
				default: 10,
				description: 'Maximum number of result chunks to return',
				displayOptions: {
					show: {
						operation: ['files'],
					},
				},
			},
			{
				displayName: 'Max Tokens',
				name: 'maxTokens',
				type: 'number',
				default: 1500,
				description: 'Token budget for returned context',
				displayOptions: {
					show: {
						operation: ['files'],
					},
				},
			},
			{
				displayName: 'Max Results',
				name: 'maxResults',
				type: 'number',
				default: 10,
				description: 'Maximum number of timeline arcs to return',
				displayOptions: {
					show: {
						operation: ['timeline'],
					},
				},
			},
			{
				displayName: 'Query',
				name: 'query',
				type: 'string',
				default: '',
				placeholder: 'auth refactor',
				description: 'Optional text query to filter timeline arcs',
				displayOptions: {
					show: {
						operation: ['timeline'],
					},
				},
			},
			{
				displayName: 'Chunk ID',
				name: 'chunkId',
				type: 'string',
				default: '',
				placeholder: 'a1b2c3d4:t5',
				description: 'Chunk identifier from a search result to drill into',
				displayOptions: {
					show: {
						operation: ['context'],
					},
				},
			},
			{
				displayName: 'Cluster ID',
				name: 'clusterId',
				type: 'string',
				default: '',
				placeholder: 'clust-abcd1234',
				description: 'Cluster identifier to show all member chunks (takes precedence over Chunk ID)',
				displayOptions: {
					show: {
						operation: ['context'],
					},
				},
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const credentials = await this.getCredentials('synaptApi');
			const serverUrl = credentials.serverUrl as string;
			const operation = this.getNodeParameter('operation', i) as string;

			let tool: string;
			let args: Record<string, unknown>;

			switch (operation) {
				case 'files': {
					const pattern = this.getNodeParameter('pattern', i) as string;
					const maxChunks = this.getNodeParameter('maxChunks', i) as number;
					const maxTokens = this.getNodeParameter('maxTokens', i) as number;
					tool = 'recall_files';
					args = { pattern, max_chunks: maxChunks, max_tokens: maxTokens };
					break;
				}
				case 'timeline': {
					const maxResults = this.getNodeParameter('maxResults', i) as number;
					const query = this.getNodeParameter('query', i) as string;
					tool = 'recall_timeline';
					args = { max_results: maxResults, ...(query ? { query } : {}) };
					break;
				}
				default: {
					const chunkId = this.getNodeParameter('chunkId', i) as string;
					const clusterId = this.getNodeParameter('clusterId', i) as string;
					tool = 'recall_context';
					args = {
						...(clusterId ? { cluster_id: clusterId } : {}),
						...(chunkId ? { chunk_id: chunkId } : {}),
					};
				}
			}

			const response = await this.helpers.httpRequest({
				method: 'POST',
				url: `${serverUrl}/mcp/call`,
				body: { tool, arguments: args },
				headers: {
					'Content-Type': 'application/json',
					...(credentials.apiKey ? { Authorization: `Bearer ${credentials.apiKey}` } : {}),
				},
			});

			results.push({ json: response });
		}

		return [results];
	}
}
