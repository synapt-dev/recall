import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';

export class SynaptSearch implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Synapt Search',
		name: 'synaptSearch',
		icon: 'file:synapt.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["query"]}}',
		description: 'Search recall memory for past sessions, decisions, and context',
		defaults: {
			name: 'Synapt Search',
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
				displayName: 'Query',
				name: 'query',
				type: 'string',
				default: '',
				required: true,
				placeholder: 'how did we fix the auth bug',
				description: 'Semantic search query against recall memory',
			},
			{
				displayName: 'Max Chunks',
				name: 'maxChunks',
				type: 'number',
				default: 5,
				description: 'Maximum number of context chunks to return',
			},
			{
				displayName: 'Max Tokens',
				name: 'maxTokens',
				type: 'number',
				default: 1500,
				description: 'Token budget for returned context',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const credentials = await this.getCredentials('synaptApi');
			const serverUrl = credentials.serverUrl as string;
			const query = this.getNodeParameter('query', i) as string;
			const maxChunks = this.getNodeParameter('maxChunks', i) as number;
			const maxTokens = this.getNodeParameter('maxTokens', i) as number;

			const response = await this.helpers.httpRequest({
				method: 'POST',
				url: `${serverUrl}/mcp/call`,
				body: {
					tool: 'recall_search',
					arguments: {
						query,
						max_chunks: maxChunks,
						max_tokens: maxTokens,
					},
				},
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
