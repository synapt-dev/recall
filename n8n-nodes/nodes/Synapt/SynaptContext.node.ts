import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';

export class SynaptContext implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Synapt Context',
		name: 'synaptContext',
		icon: 'file:synapt.svg',
		group: ['transform'],
		version: 1,
		description: 'Retrieve project context, file history, or timeline from recall memory',
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
				default: 'context',
				noDataExpression: true,
				options: [
					{
						name: 'Get Context',
						value: 'context',
						description: 'Retrieve recent project context and session summaries',
					},
					{
						name: 'File History',
						value: 'files',
						description: 'Get the history of a specific file across sessions',
					},
					{
						name: 'Timeline',
						value: 'timeline',
						description: 'View a timeline of recent recall activity',
					},
				],
			},
			{
				displayName: 'File Path',
				name: 'filePath',
				type: 'string',
				default: '',
				required: true,
				placeholder: 'src/auth.py',
				description: 'Path to the file to look up',
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
				default: 2000,
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
			const operation = this.getNodeParameter('operation', i) as string;
			const maxTokens = this.getNodeParameter('maxTokens', i) as number;

			let tool: string;
			let args: Record<string, unknown>;

			switch (operation) {
				case 'files': {
					const filePath = this.getNodeParameter('filePath', i) as string;
					tool = 'recall_files';
					args = { path: filePath, max_tokens: maxTokens };
					break;
				}
				case 'timeline':
					tool = 'recall_timeline';
					args = { max_tokens: maxTokens };
					break;
				default:
					tool = 'recall_context';
					args = { max_tokens: maxTokens };
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
