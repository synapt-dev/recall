import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';

// TODO: synapt server currently exposes stdio MCP transport only.
// These nodes are scaffolded against a planned HTTP transport endpoint.
// See: https://github.com/synapt-dev/recall/issues for tracking.

export class SynaptSave implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Synapt Save',
		name: 'synaptSave',
		icon: 'file:synapt.svg',
		group: ['output'],
		version: 1,
		subtitle: '={{$parameter["category"]}}',
		description: 'Save a fact, decision, or convention as a durable recall knowledge node',
		defaults: {
			name: 'Synapt Save',
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
				displayName: 'Content',
				name: 'content',
				type: 'string',
				default: '',
				required: true,
				typeOptions: { rows: 4 },
				placeholder: 'Always use UTC timestamps in API responses',
				description: 'The knowledge to persist in recall',
			},
			{
				displayName: 'Category',
				name: 'category',
				type: 'options',
				default: 'workflow',
				options: [
					{ name: 'Convention', value: 'convention' },
					{ name: 'Decision', value: 'decision' },
					{ name: 'Fact', value: 'fact' },
					{ name: 'Workflow', value: 'workflow' },
				],
				description: 'Knowledge category for retrieval filtering',
			},
			{
				displayName: 'Confidence',
				name: 'confidence',
				type: 'number',
				default: 0.8,
				typeOptions: {
					minValue: 0,
					maxValue: 1,
					numberStepSize: 0.1,
				},
				description: 'Confidence score for the knowledge node (0.0 to 1.0)',
			},
			{
				displayName: 'Tags',
				name: 'tags',
				type: 'string',
				default: '',
				placeholder: 'api, auth, deployment',
				description: 'Comma-separated tags for the knowledge node',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const results: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const credentials = await this.getCredentials('synaptApi');
			const serverUrl = credentials.serverUrl as string;
			const content = this.getNodeParameter('content', i) as string;
			const category = this.getNodeParameter('category', i) as string;
			const confidence = this.getNodeParameter('confidence', i) as number;
			const tagsRaw = this.getNodeParameter('tags', i) as string;
			const tags = tagsRaw ? tagsRaw.split(',').map((t) => t.trim()).filter(Boolean) : null;

			const response = await this.helpers.httpRequest({
				method: 'POST',
				url: `${serverUrl}/mcp/call`,
				body: {
					tool: 'recall_save',
					arguments: {
						content,
						category,
						confidence,
						...(tags ? { tags } : {}),
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
